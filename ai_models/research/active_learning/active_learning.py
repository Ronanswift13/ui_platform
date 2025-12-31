#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主动学习模块
智能样本选择

功能:
1. 不确定性采样 (Uncertainty Sampling)
2. 多样性采样 (Diversity Sampling)
3. 查询委员会 (Query-by-Committee)
4. 预期模型变化 (Expected Model Change)
5. 批量主动学习 (Batch Active Learning)

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import random

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """主动学习配置"""
    # 采样策略
    strategy: str = "uncertainty"  # uncertainty, diversity, committee, badge
    
    # 不确定性采样
    uncertainty_method: str = "entropy"  # entropy, margin, least_confidence
    
    # 批量采样
    batch_size: int = 100
    
    # 委员会配置
    committee_size: int = 5
    
    # 多样性配置
    diversity_weight: float = 0.5
    
    # 训练配置
    epochs_per_round: int = 10
    learning_rate: float = 1e-3
    
    # 标注预算
    total_budget: int = 1000
    initial_labeled: int = 100


if TORCH_AVAILABLE:
    # =========================================================================
    # 采样策略
    # =========================================================================
    class UncertaintySampler:
        """不确定性采样器"""
        
        def __init__(self, method: str = "entropy"):
            self.method = method
        
        def compute_uncertainty(self, probs: np.ndarray) -> np.ndarray:
            """
            计算不确定性分数
            
            Args:
                probs: (N, C) 预测概率
            
            Returns:
                uncertainty: (N,) 不确定性分数
            """
            if self.method == "entropy":
                # 熵: -sum(p * log(p))
                uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            
            elif self.method == "margin":
                # 边缘: 最大概率 - 次大概率
                sorted_probs = np.sort(probs, axis=1)
                margin = sorted_probs[:, -1] - sorted_probs[:, -2]
                uncertainty = 1 - margin  # 边缘越小,不确定性越高
            
            elif self.method == "least_confidence":
                # 最小置信度: 1 - max(p)
                uncertainty = 1 - np.max(probs, axis=1)
            
            else:
                uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            
            return uncertainty
        
        def select(self, probs: np.ndarray, n_samples: int,
                  pool_indices: Optional[np.ndarray] = None) -> np.ndarray:
            """
            选择最不确定的样本
            
            Args:
                probs: 预测概率
                n_samples: 选择数量
                pool_indices: 候选池索引
            
            Returns:
                selected: 选中的索引
            """
            uncertainty = self.compute_uncertainty(probs)
            
            if pool_indices is not None:
                # 只考虑候选池中的样本
                pool_uncertainty = uncertainty[pool_indices]
                top_indices = np.argsort(pool_uncertainty)[-n_samples:]
                selected = pool_indices[top_indices]
            else:
                selected = np.argsort(uncertainty)[-n_samples:]
            
            return selected


    class DiversitySampler:
        """多样性采样器"""
        
        def __init__(self, method: str = "coreset"):
            self.method = method
        
        def select(self, features: np.ndarray, n_samples: int,
                  labeled_indices: Optional[np.ndarray] = None) -> np.ndarray:
            """
            选择多样化的样本
            
            Args:
                features: (N, D) 特征向量
                n_samples: 选择数量
                labeled_indices: 已标注样本索引
            
            Returns:
                selected: 选中的索引
            """
            if self.method == "coreset":
                return self._coreset_selection(features, n_samples, labeled_indices)
            elif self.method == "kmeans":
                return self._kmeans_selection(features, n_samples)
            else:
                return self._coreset_selection(features, n_samples, labeled_indices)
        
        def _coreset_selection(self, features: np.ndarray, n_samples: int,
                              labeled_indices: Optional[np.ndarray] = None) -> np.ndarray:
            """
            CoreSet选择: 贪婪地选择距离已选样本最远的点
            """
            N = len(features)
            selected = []
            
            if labeled_indices is not None and len(labeled_indices) > 0:
                # 计算到已标注样本的距离
                labeled_features = features[labeled_indices]
                min_distances = np.min(cdist(features, labeled_features), axis=1)
            else:
                # 随机选择第一个
                first = np.random.randint(N)
                selected.append(first)
                min_distances = cdist(features, features[first:first+1]).squeeze()
            
            # 贪婪选择
            pool = set(range(N))
            if labeled_indices is not None:
                pool -= set(labeled_indices)
            pool -= set(selected)
            
            while len(selected) < n_samples and pool:
                # 选择距离最远的点
                pool_list = list(pool)
                pool_distances = min_distances[pool_list]
                idx = pool_list[np.argmax(pool_distances)]
                
                selected.append(idx)
                pool.remove(idx)
                
                # 更新最小距离
                new_distances = cdist(features, features[idx:idx+1]).squeeze()
                min_distances = np.minimum(min_distances, new_distances)
            
            return np.array(selected)
        
        def _kmeans_selection(self, features: np.ndarray, n_samples: int) -> np.ndarray:
            """
            K-Means选择: 选择每个聚类中心最近的样本
            """
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            kmeans.fit(features)
            
            selected = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(features - center, axis=1)
                idx = np.argmin(distances)
                if idx not in selected:
                    selected.append(idx)
            
            # 如果不够,补充最近的
            while len(selected) < n_samples:
                remaining = set(range(len(features))) - set(selected)
                if not remaining:
                    break
                selected.append(list(remaining)[0])
            
            return np.array(selected)


    class QueryByCommittee:
        """查询委员会采样"""
        
        def __init__(self, committee_size: int = 5):
            self.committee_size = committee_size
            self.committee: List[nn.Module] = []
        
        def build_committee(self, model_class, model_kwargs: Dict,
                           train_loader: DataLoader, criterion, 
                           optimizer_class, optimizer_kwargs: Dict,
                           num_epochs: int = 5):
            """
            构建委员会模型
            """
            self.committee = []
            
            for i in range(self.committee_size):
                logger.info(f"训练委员会成员 {i+1}/{self.committee_size}")
                
                # 创建模型
                model = model_class(**model_kwargs)
                optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
                
                # 随机初始化 + 训练
                model.train()
                for epoch in range(num_epochs):
                    for batch in train_loader:
                        if isinstance(batch, (tuple, list)):
                            x, y = batch
                        else:
                            x = batch
                            y = None
                        
                        optimizer.zero_grad()
                        output = model(x)
                        
                        if y is not None:
                            loss = criterion(output, y)
                        else:
                            loss = criterion(output)
                        
                        loss.backward()
                        optimizer.step()
                
                self.committee.append(model)
        
        def compute_disagreement(self, x: torch.Tensor) -> np.ndarray:
            """
            计算委员会分歧
            
            Args:
                x: 输入数据
            
            Returns:
                disagreement: (N,) 分歧分数
            """
            predictions = []
            
            for model in self.committee:
                model.eval()
                with torch.no_grad():
                    pred = F.softmax(model(x), dim=-1)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.stack(predictions)  # (C, N, K)
            
            # 投票熵
            mean_pred = predictions.mean(axis=0)  # (N, K)
            vote_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
            
            # 平均条件熵
            cond_entropy = -np.mean(
                np.sum(predictions * np.log(predictions + 1e-10), axis=2),
                axis=0
            )
            
            # 分歧 = 投票熵 - 平均条件熵
            disagreement = vote_entropy - cond_entropy
            
            return disagreement
        
        def select(self, data_loader: DataLoader, n_samples: int,
                  pool_indices: Optional[np.ndarray] = None) -> np.ndarray:
            """选择最有分歧的样本"""
            all_disagreements = []
            
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                disagreement = self.compute_disagreement(x)
                all_disagreements.append(disagreement)
            
            all_disagreements = np.concatenate(all_disagreements)
            
            if pool_indices is not None:
                pool_disagreement = all_disagreements[pool_indices]
                top_indices = np.argsort(pool_disagreement)[-n_samples:]
                selected = pool_indices[top_indices]
            else:
                selected = np.argsort(all_disagreements)[-n_samples:]
            
            return selected


    class BADGESampler:
        """
        BADGE: Batch Active Learning by Diverse Gradient Embeddings
        结合不确定性和多样性
        """
        
        def __init__(self):
            self.gradient_embeddings = None
        
        def compute_gradient_embeddings(self, model: nn.Module,
                                       data_loader: DataLoader) -> np.ndarray:
            """
            计算梯度嵌入
            """
            model.eval()
            all_embeddings = []
            
            # 获取最后一层
            last_layer = None
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    last_layer = module
            
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                x.requires_grad = True
                
                # 前向传播
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                
                # 预测标签
                pseudo_labels = logits.argmax(dim=1)
                
                # 计算梯度
                batch_embeddings = []
                for i in range(len(x)):
                    model.zero_grad()
                    loss = F.cross_entropy(logits[i:i+1], pseudo_labels[i:i+1])
                    loss.backward(retain_graph=True)
                    
                    if last_layer is not None and last_layer.weight.grad is not None:
                        grad = last_layer.weight.grad.flatten().cpu().numpy()
                        batch_embeddings.append(grad)
                
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings) if all_embeddings else np.array([])
        
        def select(self, model: nn.Module, data_loader: DataLoader,
                  n_samples: int, pool_indices: Optional[np.ndarray] = None) -> np.ndarray:
            """
            BADGE选择
            """
            # 计算梯度嵌入
            embeddings = self.compute_gradient_embeddings(model, data_loader)
            
            if len(embeddings) == 0:
                # 回退到随机选择
                if pool_indices is not None:
                    return np.random.choice(pool_indices, n_samples, replace=False)
                else:
                    return np.random.choice(len(data_loader.dataset), n_samples, replace=False)
            
            if pool_indices is not None:
                embeddings = embeddings[pool_indices]
            
            # K-Means++初始化选择
            selected = self._kmeans_pp_selection(embeddings, n_samples)
            
            if pool_indices is not None:
                selected = pool_indices[selected]
            
            return selected
        
        def _kmeans_pp_selection(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
            """K-Means++风格的选择"""
            N = len(embeddings)
            selected = []
            
            # 随机选择第一个
            first = np.random.randint(N)
            selected.append(first)
            
            # 计算距离
            min_distances = np.full(N, np.inf)
            
            while len(selected) < n_samples:
                # 更新距离
                last_selected = selected[-1]
                distances = np.linalg.norm(
                    embeddings - embeddings[last_selected], axis=1
                )
                min_distances = np.minimum(min_distances, distances)
                
                # 按距离平方采样
                probs = min_distances ** 2
                probs[selected] = 0
                probs = probs / probs.sum()
                
                # 选择下一个
                next_idx = np.random.choice(N, p=probs)
                selected.append(next_idx)
            
            return np.array(selected)


    # =========================================================================
    # 主动学习管理器
    # =========================================================================
    class ActiveLearningManager:
        """主动学习管理器"""
        
        def __init__(self, config: ActiveLearningConfig):
            self.config = config
            
            # 采样器
            self.uncertainty_sampler = UncertaintySampler(config.uncertainty_method)
            self.diversity_sampler = DiversitySampler()
            self.qbc = QueryByCommittee(config.committee_size)
            self.badge = BADGESampler()
            
            # 状态
            self.labeled_indices: List[int] = []
            self.unlabeled_indices: List[int] = []
            self.query_history: List[Dict] = []
        
        def initialize(self, dataset_size: int, initial_indices: Optional[List[int]] = None):
            """
            初始化标注状态
            
            Args:
                dataset_size: 数据集大小
                initial_indices: 初始标注样本索引
            """
            if initial_indices is not None:
                self.labeled_indices = list(initial_indices)
            else:
                # 随机选择初始样本
                all_indices = list(range(dataset_size))
                self.labeled_indices = random.sample(
                    all_indices, min(self.config.initial_labeled, dataset_size)
                )
            
            self.unlabeled_indices = [
                i for i in range(dataset_size) if i not in self.labeled_indices
            ]
            
            logger.info(f"初始化完成: {len(self.labeled_indices)} 标注, "
                       f"{len(self.unlabeled_indices)} 未标注")
        
        def query(self, model: nn.Module, dataset: Dataset,
                 n_samples: Optional[int] = None) -> np.ndarray:
            """
            查询要标注的样本
            
            Args:
                model: 当前模型
                dataset: 完整数据集
                n_samples: 查询数量
            
            Returns:
                selected: 选中的索引
            """
            if n_samples is None:
                n_samples = self.config.batch_size
            
            n_samples = min(n_samples, len(self.unlabeled_indices))
            
            if n_samples == 0:
                logger.warning("没有未标注样本可查询")
                return np.array([])
            
            # 创建未标注数据加载器
            pool_dataset = Subset(dataset, self.unlabeled_indices)
            pool_loader = DataLoader(pool_dataset, batch_size=32, shuffle=False)
            
            # 获取预测
            model.eval()
            all_probs = []
            all_features = []
            
            for batch in pool_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
                    
                    # 获取特征 (倒数第二层输出)
                    # 简化: 使用logits作为特征
                    all_features.append(logits.cpu().numpy())
            
            all_probs = np.concatenate(all_probs)
            all_features = np.concatenate(all_features)
            
            # 根据策略选择
            if self.config.strategy == "uncertainty":
                selected_pool_idx = self.uncertainty_sampler.select(all_probs, n_samples)
            
            elif self.config.strategy == "diversity":
                labeled_features = self._get_features(model, dataset, self.labeled_indices)
                selected_pool_idx = self.diversity_sampler.select(
                    all_features, n_samples, 
                    labeled_indices=np.arange(len(labeled_features)) if labeled_features is not None else None
                )
            
            elif self.config.strategy == "hybrid":
                # 混合策略: 不确定性 + 多样性
                uncertainty = self.uncertainty_sampler.compute_uncertainty(all_probs)
                
                # 先选高不确定性候选
                n_candidates = min(n_samples * 3, len(all_probs))
                candidates = np.argsort(uncertainty)[-n_candidates:]
                
                # 从候选中选择多样化样本
                candidate_features = all_features[candidates]
                diversity_idx = self.diversity_sampler.select(candidate_features, n_samples)
                selected_pool_idx = candidates[diversity_idx]
            
            elif self.config.strategy == "badge":
                selected_pool_idx = self.badge.select(
                    model, pool_loader, n_samples
                )
            
            else:
                # 随机选择
                selected_pool_idx = np.random.choice(
                    len(self.unlabeled_indices), n_samples, replace=False
                )
            
            # 映射回原始索引
            selected = np.array([self.unlabeled_indices[i] for i in selected_pool_idx])
            
            # 记录历史
            self.query_history.append({
                "round": len(self.query_history),
                "n_samples": n_samples,
                "selected": selected.tolist()
            })
            
            return selected
        
        def _get_features(self, model: nn.Module, dataset: Dataset,
                         indices: List[int]) -> Optional[np.ndarray]:
            """获取指定样本的特征"""
            if not indices:
                return None
            
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=32, shuffle=False)
            
            model.eval()
            features = []
            
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                with torch.no_grad():
                    feat = model(x)
                    features.append(feat.cpu().numpy())
            
            return np.concatenate(features) if features else None
        
        def update(self, selected_indices: np.ndarray):
            """
            更新标注状态
            
            Args:
                selected_indices: 新标注的样本索引
            """
            for idx in selected_indices:
                if idx in self.unlabeled_indices:
                    self.unlabeled_indices.remove(idx)
                    self.labeled_indices.append(idx)
            
            logger.info(f"更新后: {len(self.labeled_indices)} 标注, "
                       f"{len(self.unlabeled_indices)} 未标注")
        
        def get_labeled_subset(self, dataset: Dataset) -> Subset:
            """获取已标注子集"""
            return Subset(dataset, self.labeled_indices)
        
        def get_unlabeled_subset(self, dataset: Dataset) -> Subset:
            """获取未标注子集"""
            return Subset(dataset, self.unlabeled_indices)


    # =========================================================================
    # 主动学习训练循环
    # =========================================================================
    class ActiveLearningTrainer:
        """主动学习训练器"""
        
        def __init__(self, config: ActiveLearningConfig,
                     model_class, model_kwargs: Dict,
                     dataset: Dataset,
                     criterion, optimizer_class, optimizer_kwargs: Dict):
            self.config = config
            self.model_class = model_class
            self.model_kwargs = model_kwargs
            self.dataset = dataset
            self.criterion = criterion
            self.optimizer_class = optimizer_class
            self.optimizer_kwargs = optimizer_kwargs
            
            self.manager = ActiveLearningManager(config)
            self.model = None
            self.training_history: List[Dict] = []
        
        def train_model(self, train_loader: DataLoader) -> float:
            """训练模型一轮"""
            if self.model is None:
                self.model = self.model_class(**self.model_kwargs)
            
            optimizer = self.optimizer_class(
                self.model.parameters(), **self.optimizer_kwargs
            )
            
            self.model.train()
            total_loss = 0
            
            for epoch in range(self.config.epochs_per_round):
                epoch_loss = 0
                for batch in train_loader:
                    if isinstance(batch, (tuple, list)):
                        x, y = batch
                    else:
                        x = batch
                        y = None
                    
                    optimizer.zero_grad()
                    output = self.model(x)
                    
                    if y is not None:
                        loss = self.criterion(output, y)
                    else:
                        loss = self.criterion(output)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss = epoch_loss / len(train_loader)
            
            return total_loss
        
        def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
            """评估模型"""
            if self.model is None:
                return {"accuracy": 0.0}
            
            self.model.eval()
            correct = 0
            total = 0
            
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                else:
                    continue
                
                with torch.no_grad():
                    output = self.model(x)
                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            
            return {"accuracy": correct / total if total > 0 else 0.0}
        
        def run(self, test_loader: DataLoader, num_rounds: int = 10) -> List[Dict]:
            """
            运行主动学习循环
            
            Args:
                test_loader: 测试数据加载器
                num_rounds: 主动学习轮数
            
            Returns:
                训练历史
            """
            # 初始化
            self.manager.initialize(len(self.dataset))
            
            for round_idx in range(num_rounds):
                logger.info(f"\n=== 主动学习轮次 {round_idx + 1}/{num_rounds} ===")
                
                # 获取已标注数据
                labeled_subset = self.manager.get_labeled_subset(self.dataset)
                train_loader = DataLoader(
                    labeled_subset, batch_size=32, shuffle=True
                )
                
                # 训练模型
                train_loss = self.train_model(train_loader)
                
                # 评估
                metrics = self.evaluate(test_loader)
                
                logger.info(f"标注样本: {len(self.manager.labeled_indices)}")
                logger.info(f"训练损失: {train_loss:.4f}")
                logger.info(f"测试准确率: {metrics['accuracy']:.4f}")
                
                # 记录历史
                self.training_history.append({
                    "round": round_idx,
                    "n_labeled": len(self.manager.labeled_indices),
                    "train_loss": train_loss,
                    **metrics
                })
                
                # 检查预算
                if len(self.manager.labeled_indices) >= self.config.total_budget:
                    logger.info("达到标注预算上限")
                    break
                
                # 查询新样本
                if round_idx < num_rounds - 1:
                    selected = self.manager.query(self.model, self.dataset)
                    
                    if len(selected) > 0:
                        # 模拟标注 (实际应用中需要人工标注)
                        self.manager.update(selected)
                    else:
                        logger.info("没有更多样本可查询")
                        break
            
            return self.training_history


# =============================================================================
# 入口函数
# =============================================================================
def create_active_learner(config: Optional[ActiveLearningConfig] = None) -> ActiveLearningManager:
    """创建主动学习管理器"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = ActiveLearningConfig()
    
    return ActiveLearningManager(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if TORCH_AVAILABLE:
        # 示例
        class SimpleModel(nn.Module):
            def __init__(self, input_dim=10, num_classes=5):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x)
        
        # 创建虚拟数据集
        class DummyDataset(Dataset):
            def __init__(self, size=1000):
                self.data = torch.randn(size, 10)
                self.labels = torch.randint(0, 5, (size,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = DummyDataset(1000)
        test_dataset = DummyDataset(200)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=50,
            total_budget=500
        )
        
        trainer = ActiveLearningTrainer(
            config=config,
            model_class=SimpleModel,
            model_kwargs={"input_dim": 10, "num_classes": 5},
            dataset=dataset,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-3}
        )
        
        history = trainer.run(test_loader, num_rounds=5)
        
        print("\n训练历史:")
        for record in history:
            print(f"轮次 {record['round']}: "
                  f"标注={record['n_labeled']}, "
                  f"准确率={record['accuracy']:.4f}")
