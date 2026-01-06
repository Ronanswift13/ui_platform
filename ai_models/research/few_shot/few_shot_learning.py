#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小样本学习研究模块
新缺陷类型快速适配

功能:
1. 原型网络 (Prototypical Networks)
2. 匹配网络 (Matching Networks)
3. MAML元学习
4. 关系网络 (Relation Network)
5. 数据增强策略

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Sampler
    from torch.optim import Adam
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
class FewShotConfig:
    """小样本学习配置"""
    # 任务配置
    n_way: int = 5  # 类别数
    k_shot: int = 5  # 每类支持集样本数
    q_query: int = 15  # 每类查询集样本数
    
    # 模型配置
    backbone: str = "conv4"  # conv4, resnet12, resnet18
    feature_dim: int = 64
    hidden_dim: int = 64
    
    # 训练配置
    num_episodes: int = 10000
    learning_rate: float = 1e-3
    meta_lr: float = 1e-3  # MAML外循环学习率
    inner_lr: float = 0.01  # MAML内循环学习率
    inner_steps: int = 5  # MAML内循环步数
    
    # 数据增强
    use_augmentation: bool = True
    
    # 保存配置
    save_dir: str = "checkpoints/few_shot"
    save_freq: int = 500


# =============================================================================
# 骨干网络
# =============================================================================
if TORCH_AVAILABLE:
    
    class ConvBlock(nn.Module):
        """卷积块"""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d(2)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(F.relu(self.bn(self.conv(x))))
    
    
    class Conv4Backbone(nn.Module):
        """4层卷积骨干网络"""
        
        def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
            super().__init__()
            
            self.layer1 = ConvBlock(in_channels, hidden_dim)
            self.layer2 = ConvBlock(hidden_dim, hidden_dim)
            self.layer3 = ConvBlock(hidden_dim, hidden_dim)
            self.layer4 = ConvBlock(hidden_dim, hidden_dim)
            
            self.feature_dim = hidden_dim
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x.view(x.size(0), -1)
    
    
    class ResBlock(nn.Module):
        """残差块"""
        
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)
    
    
    class ResNet12Backbone(nn.Module):
        """ResNet12骨干网络"""
        
        def __init__(self, in_channels: int = 3):
            super().__init__()
            
            self.layer1 = self._make_layer(in_channels, 64)
            self.layer2 = self._make_layer(64, 128)
            self.layer3 = self._make_layer(128, 256)
            self.layer4 = self._make_layer(256, 512)
            
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = 512
        
        def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                ResBlock(in_channels, out_channels),
                ResBlock(out_channels, out_channels),
                ResBlock(out_channels, out_channels),
                nn.MaxPool2d(2)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.pool(x)
            return x.view(x.size(0), -1)


    # =========================================================================
    # 原型网络
    # =========================================================================
    class PrototypicalNetwork(nn.Module):
        """
        原型网络 (Prototypical Networks)
        
        计算每个类别的原型(特征均值),
        使用到原型的距离进行分类
        """
        
        def __init__(self, config: FewShotConfig):
            super().__init__()
            
            if config.backbone == "conv4":
                self.encoder = Conv4Backbone(hidden_dim=config.hidden_dim)
            else:
                self.encoder = ResNet12Backbone()
            
            self.config = config
        
        def compute_prototypes(self, 
                               support: torch.Tensor,
                               n_way: int,
                               k_shot: int) -> torch.Tensor:
            """
            计算类别原型
            
            Args:
                support: (n_way * k_shot, feature_dim) 支持集特征
                n_way: 类别数
                k_shot: 每类样本数
            
            Returns:
                prototypes: (n_way, feature_dim) 原型
            """
            # 重塑为 (n_way, k_shot, feature_dim)
            support = support.view(n_way, k_shot, -1)
            
            # 计算均值作为原型
            prototypes = support.mean(dim=1)
            
            return prototypes
        
        def forward(self, 
                    support: torch.Tensor,
                    query: torch.Tensor,
                    n_way: int,
                    k_shot: int) -> torch.Tensor:
            """
            Args:
                support: (n_way * k_shot, C, H, W) 支持集
                query: (n_query, C, H, W) 查询集
                n_way: 类别数
                k_shot: 每类支持集样本数
            
            Returns:
                logits: (n_query, n_way) 分类logits
            """
            # 编码
            support_features = self.encoder(support)  # (n_way * k_shot, D)
            query_features = self.encoder(query)  # (n_query, D)
            
            # 计算原型
            prototypes = self.compute_prototypes(support_features, n_way, k_shot)
            
            # 计算欧氏距离
            dists = torch.cdist(query_features, prototypes)  # (n_query, n_way)
            
            # 负距离作为logits
            logits = -dists
            
            return logits


    # =========================================================================
    # 匹配网络
    # =========================================================================
    class MatchingNetwork(nn.Module):
        """
        匹配网络 (Matching Networks)
        
        使用注意力机制计算查询样本与支持集的相似度
        """
        
        def __init__(self, config: FewShotConfig):
            super().__init__()
            
            if config.backbone == "conv4":
                self.encoder = Conv4Backbone(hidden_dim=config.hidden_dim)
            else:
                self.encoder = ResNet12Backbone()
            
            feature_dim = self.encoder.feature_dim
            
            # Full Context Embeddings (FCE)
            self.support_lstm = nn.LSTM(
                feature_dim, feature_dim // 2, 
                bidirectional=True, batch_first=True
            )
            self.query_lstm = nn.LSTM(
                feature_dim, feature_dim,
                batch_first=True
            )
            
            self.config = config
        
        def forward(self,
                    support: torch.Tensor,
                    query: torch.Tensor,
                    support_labels: torch.Tensor,
                    n_way: int,
                    k_shot: int) -> torch.Tensor:
            """
            Args:
                support: (n_way * k_shot, C, H, W)
                query: (n_query, C, H, W)
                support_labels: (n_way * k_shot,) 支持集标签
                n_way: 类别数
                k_shot: 每类样本数
            
            Returns:
                logits: (n_query, n_way)
            """
            # 编码
            support_features = self.encoder(support)  # (N, D)
            query_features = self.encoder(query)  # (Q, D)
            
            N = support_features.size(0)
            Q = query_features.size(0)
            
            # FCE for support
            support_features = support_features.unsqueeze(0)  # (1, N, D)
            support_features, _ = self.support_lstm(support_features)
            support_features = support_features.squeeze(0)  # (N, D)
            
            # 计算注意力 (余弦相似度)
            support_features = F.normalize(support_features, dim=1)
            query_features = F.normalize(query_features, dim=1)
            
            attention = torch.mm(query_features, support_features.t())  # (Q, N)
            attention = F.softmax(attention, dim=1)
            
            # 将注意力分配到类别
            # 创建one-hot标签矩阵
            labels_onehot = F.one_hot(support_labels, n_way).float()  # (N, n_way)
            
            # 加权求和
            logits = torch.mm(attention, labels_onehot)  # (Q, n_way)
            
            return logits


    # =========================================================================
    # 关系网络
    # =========================================================================
    class RelationNetwork(nn.Module):
        """
        关系网络 (Relation Network)
        
        学习一个关系函数来比较查询样本和支持集
        """
        
        def __init__(self, config: FewShotConfig):
            super().__init__()
            
            if config.backbone == "conv4":
                self.encoder = Conv4Backbone(hidden_dim=config.hidden_dim)
            else:
                self.encoder = ResNet12Backbone()
            
            feature_dim = config.hidden_dim * 2  # 连接两个特征
            
            # 关系模块
            self.relation_module = nn.Sequential(
                nn.Linear(feature_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            self.config = config
        
        def forward(self,
                    support: torch.Tensor,
                    query: torch.Tensor,
                    n_way: int,
                    k_shot: int) -> torch.Tensor:
            """
            Args:
                support: (n_way * k_shot, C, H, W)
                query: (n_query, C, H, W)
                n_way: 类别数
                k_shot: 每类样本数
            
            Returns:
                relations: (n_query, n_way) 关系分数
            """
            # 编码
            support_features = self.encoder(support)  # (N, D)
            query_features = self.encoder(query)  # (Q, D)
            
            N = support_features.size(0)
            Q = query_features.size(0)
            D = support_features.size(1)
            
            # 计算原型
            support_features = support_features.view(n_way, k_shot, D)
            prototypes = support_features.mean(dim=1)  # (n_way, D)
            
            # 扩展并连接
            query_expanded = query_features.unsqueeze(1).expand(-1, n_way, -1)  # (Q, n_way, D)
            proto_expanded = prototypes.unsqueeze(0).expand(Q, -1, -1)  # (Q, n_way, D)
            
            concat_features = torch.cat([query_expanded, proto_expanded], dim=2)  # (Q, n_way, 2D)
            concat_features = concat_features.view(Q * n_way, -1)
            
            # 计算关系分数
            relations = self.relation_module(concat_features)
            relations = relations.view(Q, n_way)
            
            return relations


    # =========================================================================
    # MAML元学习
    # =========================================================================
    class MAML(nn.Module):
        """
        Model-Agnostic Meta-Learning (MAML)
        
        通过元学习找到好的初始化参数
        """
        
        def __init__(self, config: FewShotConfig):
            super().__init__()
            
            if config.backbone == "conv4":
                self.encoder = Conv4Backbone(hidden_dim=config.hidden_dim)
            else:
                self.encoder = ResNet12Backbone()
            
            self.classifier = nn.Linear(self.encoder.feature_dim, config.n_way)
            
            self.inner_lr = config.inner_lr
            self.inner_steps = config.inner_steps
            self.config = config
        
        def clone_module(self, module: nn.Module) -> nn.Module:
            """克隆模块和参数"""
            clone = copy.deepcopy(module)
            return clone
        
        def adapt(self, 
                  support: torch.Tensor,
                  support_labels: torch.Tensor) -> Tuple[nn.Module, nn.Module]:
            """
            内循环适应
            
            Args:
                support: 支持集
                support_labels: 支持集标签
            
            Returns:
                adapted_encoder, adapted_classifier
            """
            # 克隆参数
            adapted_encoder = self.clone_module(self.encoder)
            adapted_classifier = self.clone_module(self.classifier)
            
            # 内循环优化
            for _ in range(self.inner_steps):
                features = adapted_encoder(support)
                logits = adapted_classifier(features)
                loss = F.cross_entropy(logits, support_labels)
                
                # 计算梯度
                encoder_grads = torch.autograd.grad(
                    loss, adapted_encoder.parameters(), 
                    create_graph=True, retain_graph=True
                )
                classifier_grads = torch.autograd.grad(
                    loss, adapted_classifier.parameters(),
                    create_graph=True
                )
                
                # 更新参数
                for param, grad in zip(adapted_encoder.parameters(), encoder_grads):
                    param.data = param.data - self.inner_lr * grad
                
                for param, grad in zip(adapted_classifier.parameters(), classifier_grads):
                    param.data = param.data - self.inner_lr * grad
            
            return adapted_encoder, adapted_classifier
        
        def forward(self,
                    support: torch.Tensor,
                    query: torch.Tensor,
                    support_labels: torch.Tensor,
                    query_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                support: 支持集
                query: 查询集
                support_labels: 支持集标签
                query_labels: 查询集标签 (训练时需要)
            
            Returns:
                logits, loss
            """
            # 适应
            adapted_encoder, adapted_classifier = self.adapt(support, support_labels)
            
            # 在查询集上评估
            features = adapted_encoder(query)
            logits = adapted_classifier(features)
            
            if query_labels is not None:
                loss = F.cross_entropy(logits, query_labels)
            else:
                loss = torch.tensor(0.0)
            
            return logits, loss


    # =========================================================================
    # Episode采样器
    # =========================================================================
    class EpisodeSampler(Sampler):
        """Episode采样器"""
        
        def __init__(self,
                     labels: List[int],
                     n_way: int,
                     k_shot: int,
                     q_query: int,
                     num_episodes: int):
            self.labels = np.array(labels)
            self.n_way = n_way
            self.k_shot = k_shot
            self.q_query = q_query
            self.num_episodes = num_episodes
            
            # 按类别组织索引
            self.class_indices = {}
            for idx, label in enumerate(labels):
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(idx)
            
            self.classes = list(self.class_indices.keys())
        
        def __len__(self) -> int:
            return self.num_episodes
        
        def __iter__(self):
            for _ in range(self.num_episodes):
                # 随机选择类别
                selected_classes = np.random.choice(
                    self.classes, self.n_way, replace=False
                )
                
                support_indices = []
                query_indices = []
                
                for cls in selected_classes:
                    cls_indices = np.array(self.class_indices[cls])
                    
                    # 随机选择样本
                    selected = np.random.choice(
                        cls_indices, 
                        self.k_shot + self.q_query,
                        replace=False
                    )
                    
                    support_indices.extend(selected[:self.k_shot])
                    query_indices.extend(selected[self.k_shot:])
                
                yield support_indices + query_indices


    # =========================================================================
    # 数据集
    # =========================================================================
    class FewShotDataset(Dataset):
        """小样本学习数据集"""
        
        def __init__(self,
                     data_root: str,
                     split: str = "train",
                     augment: bool = True):
            self.data_root = data_root
            self.split = split
            self.augment = augment
            
            # 加载数据
            self.samples = []
            self.labels = []
            self._load_data()
        
        def _load_data(self):
            """加载数据"""
            # 这里是占位实现,实际应该从文件加载
            # 生成虚拟数据
            num_classes = 20 if self.split == "train" else 5
            samples_per_class = 100
            
            for cls in range(num_classes):
                for _ in range(samples_per_class):
                    # 虚拟图像数据
                    sample = np.random.randn(3, 84, 84).astype(np.float32)
                    self.samples.append(sample)
                    self.labels.append(cls)
        
        def __len__(self) -> int:
            return len(self.samples)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            sample = torch.from_numpy(self.samples[idx])
            label = self.labels[idx]
            
            if self.augment:
                sample = self._augment(sample)
            
            return sample, label
        
        def _augment(self, x: torch.Tensor) -> torch.Tensor:
            """数据增强"""
            # 随机水平翻转
            if np.random.random() > 0.5:
                x = torch.flip(x, [2])
            
            # 随机裁剪
            if np.random.random() > 0.5:
                pad = 4
                x = F.pad(x.unsqueeze(0), [pad] * 4, mode='reflect').squeeze(0)
                h, w = x.shape[1], x.shape[2]
                new_h, new_w = h - 2 * pad, w - 2 * pad
                top = np.random.randint(0, 2 * pad)
                left = np.random.randint(0, 2 * pad)
                x = x[:, top:top + new_h, left:left + new_w]
            
            return x


    # =========================================================================
    # 训练器
    # =========================================================================
    class FewShotTrainer:
        """小样本学习训练器"""
        
        def __init__(self, 
                     config: FewShotConfig,
                     model_type: str = "prototypical"):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 创建模型
            if model_type == "prototypical":
                self.model = PrototypicalNetwork(config)
            elif model_type == "matching":
                self.model = MatchingNetwork(config)
            elif model_type == "relation":
                self.model = RelationNetwork(config)
            elif model_type == "maml":
                self.model = MAML(config)
            else:
                self.model = PrototypicalNetwork(config)
            
            self.model = self.model.to(self.device)
            self.model_type = model_type
            
            # 优化器
            self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
            
            # 数据加载器
            self.train_dataset = FewShotDataset("data/few_shot", "train")
            self.val_dataset = FewShotDataset("data/few_shot", "val", augment=False)
            
            self.train_sampler = EpisodeSampler(
                self.train_dataset.labels,
                config.n_way, config.k_shot, config.q_query,
                config.num_episodes
            )
            
            self.val_sampler = EpisodeSampler(
                self.val_dataset.labels,
                config.n_way, config.k_shot, config.q_query,
                100  # 100个验证episode
            )
            
            os.makedirs(config.save_dir, exist_ok=True)
        
        def train_episode(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
            """训练一个episode"""
            data, labels = batch
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            n_way = self.config.n_way
            k_shot = self.config.k_shot
            q_query = self.config.q_query
            
            # 分离支持集和查询集
            support_size = n_way * k_shot
            support = data[:support_size]
            query = data[support_size:]
            support_labels = labels[:support_size]
            query_labels = labels[support_size:]
            
            # 重新映射标签到0-n_way
            unique_labels = support_labels.unique()
            label_map = {l.item(): i for i, l in enumerate(unique_labels)}
            support_labels = torch.tensor([label_map[l.item()] for l in support_labels], device=self.device)
            query_labels = torch.tensor([label_map[l.item()] for l in query_labels], device=self.device)
            
            self.optimizer.zero_grad()
            
            if self.model_type == "maml":
                logits, loss = self.model(support, query, support_labels, query_labels)
            elif self.model_type == "matching":
                logits = self.model(support, query, support_labels, n_way, k_shot)
                loss = F.cross_entropy(logits, query_labels)
            else:
                logits = self.model(support, query, n_way, k_shot)
                loss = F.cross_entropy(logits, query_labels)
            
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            pred = logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean().item()
            
            return {"loss": loss.item(), "accuracy": acc}
        
        def train(self):
            """训练"""
            logger.info(f"开始训练 {self.model_type} 模型")
            logger.info(f"配置: {self.config.n_way}-way {self.config.k_shot}-shot")
            
            best_val_acc = 0.0
            
            for episode, indices in enumerate(self.train_sampler):
                # 获取batch
                data = torch.stack([self.train_dataset[i][0] for i in indices])
                labels = torch.tensor([self.train_dataset[i][1] for i in indices])
                
                # 训练
                metrics = self.train_episode((data, labels))
                
                if episode % 100 == 0:
                    logger.info(
                        f"Episode {episode}: "
                        f"Loss: {metrics['loss']:.4f}, "
                        f"Acc: {metrics['accuracy']:.4f}"
                    )
                
                # 验证
                if (episode + 1) % self.config.save_freq == 0:
                    val_acc = self.validate()
                    logger.info(f"Validation Accuracy: {val_acc:.4f}")
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        self.save_checkpoint("best.pth")
            
            logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        
        @torch.no_grad()
        def validate(self) -> float:
            """验证"""
            self.model.eval()
            
            total_acc = 0.0
            num_episodes = 0
            
            for indices in self.val_sampler:
                data = torch.stack([self.val_dataset[i][0] for i in indices])
                labels = torch.tensor([self.val_dataset[i][1] for i in indices])
                
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                n_way = self.config.n_way
                k_shot = self.config.k_shot
                
                support_size = n_way * k_shot
                support = data[:support_size]
                query = data[support_size:]
                support_labels = labels[:support_size]
                query_labels = labels[support_size:]
                
                # 重映射标签
                unique_labels = support_labels.unique()
                label_map = {l.item(): i for i, l in enumerate(unique_labels)}
                support_labels = torch.tensor([label_map[l.item()] for l in support_labels], device=self.device)
                query_labels = torch.tensor([label_map[l.item()] for l in query_labels], device=self.device)
                
                if self.model_type == "maml":
                    logits, _ = self.model(support, query, support_labels)
                elif self.model_type == "matching":
                    logits = self.model(support, query, support_labels, n_way, k_shot)
                else:
                    logits = self.model(support, query, n_way, k_shot)
                
                pred = logits.argmax(dim=1)
                acc = (pred == query_labels).float().mean().item()
                total_acc += acc
                num_episodes += 1
            
            self.model.train()
            return total_acc / num_episodes
        
        def save_checkpoint(self, filename: str):
            path = os.path.join(self.config.save_dir, filename)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__
            }, path)
            logger.info(f"保存检查点: {path}")


# =============================================================================
# 快速适配接口
# =============================================================================
def adapt_to_new_defect(model_path: str,
                        support_images: List[np.ndarray],
                        support_labels: List[int],
                        model_type: str = "prototypical") -> 'nn.Module':
    """
    快速适配新缺陷类型
    
    Args:
        model_path: 预训练模型路径
        support_images: 支持集图像列表
        support_labels: 支持集标签列表
        model_type: 模型类型
    
    Returns:
        适配后的模型
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    config = FewShotConfig()
    
    # 加载模型
    if model_type == "prototypical":
        model = PrototypicalNetwork(config)
    elif model_type == "maml":
        model = MAML(config)
    else:
        model = PrototypicalNetwork(config)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 如果是MAML,执行内循环适应
    if model_type == "maml":
        support = torch.stack([torch.from_numpy(img) for img in support_images])
        labels = torch.tensor(support_labels)
        
        adapted_encoder, adapted_classifier = model.adapt(support, labels)
        model.encoder = adapted_encoder
        model.classifier = adapted_classifier
    
    logger.info(f"模型适配完成, 支持集大小: {len(support_images)}")
    
    return model


# =============================================================================
# 入口函数
# =============================================================================
def train_few_shot(config: Optional[FewShotConfig] = None,
                   model_type: str = "prototypical"):
    """训练小样本学习模型"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = FewShotConfig()
    
    trainer = FewShotTrainer(config, model_type)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = FewShotConfig(
        n_way=5,
        k_shot=5,
        num_episodes=1000,
        save_dir="checkpoints/few_shot"
    )
    
    train_few_shot(config, model_type="prototypical")
