#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不确定性量化研究模块
贝叶斯深度学习

功能:
1. MC Dropout
2. 深度集成 (Deep Ensemble)
3. 贝叶斯神经网络 (BNN)
4. 变分推断
5. 校准方法

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.special import softmax

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class UncertaintyConfig:
    """不确定性量化配置"""
    # 模型配置
    method: str = "mc_dropout"  # mc_dropout, ensemble, bnn
    num_samples: int = 30  # MC采样次数或集成数量
    mc_samples: Optional[int] = None  # 兼容参数别名
    dropout_rate: float = 0.2
    
    # BNN配置
    prior_sigma: float = 1.0
    posterior_rho_init: float = -3.0
    kl_weight: float = 1e-6
    
    # 校准配置
    temperature: float = 1.0
    
    # 训练配置
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3

    def __post_init__(self):
        if self.mc_samples is not None:
            self.num_samples = self.mc_samples


# =============================================================================
# MC Dropout
# =============================================================================
if TORCH_AVAILABLE:
    
    class MCDropoutModel(nn.Module):
        """
        MC Dropout模型
        
        在推理时保持Dropout激活,通过多次采样估计不确定性
        """
        
        def __init__(self, 
                     base_model: nn.Module,
                     dropout_rate: float = 0.2):
            super().__init__()
            self.base_model = base_model
            self.dropout_rate = dropout_rate
            
            # 确保模型有Dropout层
            self._add_dropout_layers()
        
        def _add_dropout_layers(self):
            """在模型中添加Dropout层"""
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    # 在全连接层后添加Dropout
                    setattr(module, '_mc_dropout', nn.Dropout(self.dropout_rate))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_model(x)
        
        def predict_with_uncertainty(self, 
                                     x: torch.Tensor,
                                     num_samples: int = 30) -> Dict[str, torch.Tensor]:
            """
            带不确定性的预测
            
            Args:
                x: 输入
                num_samples: 采样次数
            
            Returns:
                dict with mean, std, predictions
            """
            self.train()  # 保持Dropout激活
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    output = self.forward(x)
                    predictions.append(output)
            
            predictions = torch.stack(predictions)  # (num_samples, batch, num_classes)
            
            # 计算统计量
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            
            # 预测熵 (不确定性度量)
            probs = F.softmax(mean, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            
            # 预测方差 (认知不确定性)
            predictive_variance = predictions.var(dim=0).mean(dim=-1)
            
            return {
                "mean": mean,
                "std": std,
                "entropy": entropy,
                "predictive_variance": predictive_variance,
                "all_predictions": predictions
            }


    # =========================================================================
    # 深度集成
    # =========================================================================
    class DeepEnsemble(nn.Module):
        """
        深度集成 (Deep Ensemble)
        
        训练多个独立的模型,通过集成估计不确定性
        """
        
        def __init__(self,
                     model_factory: callable,
                     num_models: int = 5):
            super().__init__()
            self.num_models = num_models
            self.models = nn.ModuleList([
                model_factory() for _ in range(num_models)
            ])
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """默认返回平均预测"""
            outputs = torch.stack([model(x) for model in self.models])
            return outputs.mean(dim=0)
        
        def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """带不确定性的预测"""
            self.eval()
            
            with torch.no_grad():
                outputs = torch.stack([model(x) for model in self.models])
            
            mean = outputs.mean(dim=0)
            std = outputs.std(dim=0)
            
            probs = F.softmax(mean, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            
            # 集成分歧度量
            disagreement = outputs.var(dim=0).mean(dim=-1)
            
            return {
                "mean": mean,
                "std": std,
                "entropy": entropy,
                "disagreement": disagreement,
                "all_predictions": outputs
            }
        
        def train_ensemble(self,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           num_epochs: int = 50,
                           lr: float = 1e-3):
            """训练集成模型"""
            for i, model in enumerate(self.models):
                logger.info(f"训练模型 {i+1}/{self.num_models}")
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0.0
                    
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        output = model(batch_x)
                        loss = F.cross_entropy(output, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")


    # =========================================================================
    # 贝叶斯神经网络
    # =========================================================================
    class BayesianLinear(nn.Module):
        """
        贝叶斯线性层
        
        使用变分推断学习权重分布
        """
        
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     prior_sigma: float = 1.0,
                     posterior_rho_init: float = -3.0):
            super().__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            
            # 先验
            self.prior_sigma = prior_sigma
            
            # 后验参数 (均值和log方差)
            self.weight_mu = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(0, 0.1)
            )
            self.weight_rho = nn.Parameter(
                torch.Tensor(out_features, in_features).fill_(posterior_rho_init)
            )
            
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).fill_(0))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).fill_(posterior_rho_init))
        
        def _rho_to_sigma(self, rho: torch.Tensor) -> torch.Tensor:
            """将rho转换为sigma (使用softplus)"""
            return F.softplus(rho)
        
        def _sample_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
            """从后验分布采样权重"""
            weight_sigma = self._rho_to_sigma(self.weight_rho)
            bias_sigma = self._rho_to_sigma(self.bias_rho)
            
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps
            
            return weight, bias
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
                output, kl_divergence
            """
            weight, bias = self._sample_weights()
            output = F.linear(x, weight, bias)
            
            # 计算KL散度
            weight_sigma = self._rho_to_sigma(self.weight_rho)
            bias_sigma = self._rho_to_sigma(self.bias_rho)
            
            kl = self._kl_divergence(
                self.weight_mu, weight_sigma,
                0, self.prior_sigma
            ) + self._kl_divergence(
                self.bias_mu, bias_sigma,
                0, self.prior_sigma
            )
            
            return output, kl
        
        def _kl_divergence(self,
                           mu_q: torch.Tensor,
                           sigma_q: torch.Tensor,
                           mu_p: float,
                           sigma_p: float) -> torch.Tensor:
            """计算两个高斯分布间的KL散度"""
            kl = torch.log(sigma_p / sigma_q) + \
                 (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) - 0.5
            return kl.sum()


    class BayesianNN(nn.Module):
        """
        贝叶斯神经网络
        """
        
        def __init__(self,
                     input_dim: int,
                     hidden_dim: int,
                     output_dim: int,
                     num_layers: int = 3,
                     prior_sigma: float = 1.0):
            super().__init__()
            
            layers = []
            in_dim = input_dim
            
            for i in range(num_layers - 1):
                layers.append(BayesianLinear(in_dim, hidden_dim, prior_sigma))
                in_dim = hidden_dim
            
            layers.append(BayesianLinear(in_dim, output_dim, prior_sigma))
            
            self.layers = nn.ModuleList(layers)
            self.num_layers = num_layers
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
                output, total_kl
            """
            total_kl = 0.0
            
            for i, layer in enumerate(self.layers):
                x, kl = layer(x)
                total_kl = total_kl + kl
                
                if i < self.num_layers - 1:
                    x = F.relu(x)
            
            return x, total_kl
        
        def predict_with_uncertainty(self,
                                     x: torch.Tensor,
                                     num_samples: int = 30) -> Dict[str, torch.Tensor]:
            """带不确定性的预测"""
            self.eval()
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    output, _ = self.forward(x)
                    predictions.append(output)
            
            predictions = torch.stack(predictions)
            
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            
            probs = F.softmax(mean, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            
            # 认知不确定性 (模型不确定性)
            epistemic = predictions.var(dim=0).mean(dim=-1)
            
            # 任意不确定性 (数据不确定性) - 从softmax输出估计
            aleatoric = (probs * (1 - probs)).sum(dim=-1)
            
            return {
                "mean": mean,
                "std": std,
                "entropy": entropy,
                "epistemic_uncertainty": epistemic,
                "aleatoric_uncertainty": aleatoric,
                "all_predictions": predictions
            }


    class BayesianNNTrainer:
        """贝叶斯神经网络训练器"""
        
        def __init__(self, 
                     model: BayesianNN,
                     config: UncertaintyConfig):
            self.model = model
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate
            )
        
        def train(self, train_loader: DataLoader, num_epochs: int = None):
            """训练BNN"""
            if num_epochs is None:
                num_epochs = self.config.num_epochs
            
            num_batches = len(train_loader)
            
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0.0
                total_nll = 0.0
                total_kl = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    output, kl = self.model(batch_x)
                    
                    # ELBO损失 = NLL + KL权重 * KL散度
                    nll = F.cross_entropy(output, batch_y)
                    kl_scaled = self.config.kl_weight * kl / num_batches
                    loss = nll + kl_scaled
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    total_nll += nll.item()
                    total_kl += kl.item()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}: "
                        f"Loss={total_loss/num_batches:.4f}, "
                        f"NLL={total_nll/num_batches:.4f}, "
                        f"KL={total_kl/num_batches:.4f}"
                    )


    # =========================================================================
    # 校准方法
    # =========================================================================
    class TemperatureScaling(nn.Module):
        """
        温度缩放校准
        
        学习一个温度参数来校准模型输出
        """
        
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model
            self.temperature = nn.Parameter(torch.ones(1))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.model(x)
            return logits / self.temperature
        
        def calibrate(self, val_loader: DataLoader, lr: float = 0.01, max_iter: int = 50):
            """校准温度参数"""
            self.model.eval()
            optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
            
            # 收集所有验证数据的logits和标签
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = self.model(batch_x)
                    all_logits.append(logits)
                    all_labels.append(batch_y)
            
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            
            def closure():
                optimizer.zero_grad()
                scaled_logits = all_logits / self.temperature
                loss = F.cross_entropy(scaled_logits, all_labels)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            logger.info(f"校准完成, 温度参数: {self.temperature.item():.4f}")
            
            return self.temperature.item()


    def compute_ece(probs: np.ndarray, 
                    labels: np.ndarray,
                    num_bins: int = 15) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        
        Args:
            probs: (N, C) 预测概率
            labels: (N,) 真实标签
            num_bins: 分箱数量
        
        Returns:
            ECE值
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bins = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() * np.abs(bin_acc - bin_conf)
        
        return ece / len(labels)


    def compute_reliability_diagram(probs: np.ndarray,
                                     labels: np.ndarray,
                                     num_bins: int = 15) -> Dict[str, np.ndarray]:
        """
        计算可靠性图数据
        
        Returns:
            dict with bin_accuracies, bin_confidences, bin_counts
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bins = np.linspace(0, 1, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(num_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if mask.sum() > 0:
                bin_accuracies.append(accuracies[mask].mean())
                bin_confidences.append(confidences[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bins[i] + bins[i + 1]) / 2)
                bin_counts.append(0)
        
        return {
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts),
            "bin_edges": bins
        }


# =============================================================================
# 不确定性感知决策
# =============================================================================
class UncertaintyAwareClassifier:
    """
    不确定性感知分类器
    
    根据不确定性水平决定是否拒绝预测
    """
    
    def __init__(self,
                 model: nn.Module,
                 uncertainty_method: str = "mc_dropout",
                 num_samples: int = 30,
                 rejection_threshold: float = 0.5):
        self.model = model
        self.uncertainty_method = uncertainty_method
        self.num_samples = num_samples
        self.rejection_threshold = rejection_threshold
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        带拒绝选项的预测
        
        Returns:
            dict with prediction, confidence, uncertainty, should_reject
        """
        if hasattr(self.model, 'predict_with_uncertainty'):
            results = self.model.predict_with_uncertainty(x, self.num_samples)
        else:
            # 使用MC Dropout包装
            mc_model = MCDropoutModel(self.model)
            results = mc_model.predict_with_uncertainty(x, self.num_samples)
        
        probs = F.softmax(results["mean"], dim=-1)
        confidence = probs.max(dim=-1)[0]
        prediction = probs.argmax(dim=-1)
        
        uncertainty = results["entropy"] if "entropy" in results else results["std"].mean(dim=-1)
        
        # 决定是否拒绝
        should_reject = uncertainty > self.rejection_threshold
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "should_reject": should_reject,
            "probabilities": probs
        }
    
    def set_threshold_from_validation(self,
                                       val_loader: DataLoader,
                                       target_coverage: float = 0.9):
        """
        从验证集设置拒绝阈值
        
        Args:
            val_loader: 验证数据加载器
            target_coverage: 目标覆盖率 (1 - 拒绝率)
        """
        all_uncertainties = []
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                result = self.predict(batch_x)
                all_uncertainties.append(result["uncertainty"])
        
        all_uncertainties = torch.cat(all_uncertainties).numpy()
        
        # 设置阈值使得覆盖率达到目标
        self.rejection_threshold = np.percentile(all_uncertainties, target_coverage * 100)
        
        logger.info(f"拒绝阈值设置为: {self.rejection_threshold:.4f}")
        logger.info(f"预期覆盖率: {target_coverage:.2%}")


# =============================================================================
# 入口函数
# =============================================================================
def create_uncertainty_model(base_model: 'nn.Module',
                             config: Optional[UncertaintyConfig] = None) -> nn.Module:
    """创建不确定性量化模型"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = UncertaintyConfig()
    
    if config.method == "mc_dropout":
        return MCDropoutModel(base_model, config.dropout_rate)
    elif config.method == "ensemble":
        return DeepEnsemble(lambda: type(base_model)(), config.num_samples)
    elif config.method == "bnn":
        # 需要重新构建为贝叶斯网络
        logger.warning("BNN需要从头构建,返回MC Dropout模型")
        return MCDropoutModel(base_model, config.dropout_rate)
    else:
        return MCDropoutModel(base_model, config.dropout_rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例
    config = UncertaintyConfig(
        method="mc_dropout",
        num_samples=30,
        dropout_rate=0.2
    )
    
    logger.info(f"不确定性量化配置: {config}")
