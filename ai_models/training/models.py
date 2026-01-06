#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型定义模块
============

包含变电站巡检所需的所有模型架构:
- 目标检测: YOLOv8, RT-DETR
- 语义分割: U-Net
- 分类: CNN, ResNet
- 关键点检测: HRNet
- OCR: CRNN

作者: 破夜绘明团队
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

# PyTorch导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 基础模块
# =============================================================================
class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBNSiLU(nn.Module):
    """卷积 + BN + SiLU (YOLO风格)"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SPPF(nn.Module):
    """空间金字塔池化 - Fast"""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.cv2 = ConvBNSiLU(c_ * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8风格)"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(out_channels * e)
        self.cv1 = ConvBNSiLU(in_channels, 2 * self.c, 1, 1, 0)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_channels, 1, 1, 0)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """标准Bottleneck"""
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.cv2 = ConvBNSiLU(c_, out_channels, 3, 1, 1)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# =============================================================================
# YOLOv8模型 (目标检测)
# =============================================================================
class YOLOv8Backbone(nn.Module):
    """YOLOv8骨干网络"""
    
    # 模型规模配置 [depth_multiple, width_multiple]
    SCALES = {
        'n': [0.33, 0.25],
        's': [0.33, 0.50],
        'm': [0.67, 0.75],
        'l': [1.00, 1.00],
        'x': [1.00, 1.25],
    }
    
    def __init__(self, scale='n', in_channels=3):
        super().__init__()
        
        d, w = self.SCALES[scale]
        
        # 通道数
        c1 = int(64 * w)
        c2 = int(128 * w)
        c3 = int(256 * w)
        c4 = int(512 * w)
        c5 = int(1024 * w)
        
        # Stem
        self.stem = ConvBNSiLU(in_channels, c1, 3, 2, 1)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(c1, c2, 3, 2, 1),
            C2f(c2, c2, n=max(1, int(3 * d)), shortcut=True)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNSiLU(c2, c3, 3, 2, 1),
            C2f(c3, c3, n=max(1, int(6 * d)), shortcut=True)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNSiLU(c3, c4, 3, 2, 1),
            C2f(c4, c4, n=max(1, int(6 * d)), shortcut=True)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNSiLU(c4, c5, 3, 2, 1),
            C2f(c5, c5, n=max(1, int(3 * d)), shortcut=True),
            SPPF(c5, c5)
        )
        
        self.out_channels = [c3, c4, c5]
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


class YOLOv8Neck(nn.Module):
    """YOLOv8 FPN + PAN"""
    def __init__(self, in_channels, scale='n'):
        super().__init__()
        
        d, w = YOLOv8Backbone.SCALES[scale]
        c3, c4, c5 = in_channels
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # FPN (top-down)
        self.reduce_c5 = ConvBNSiLU(c5, c4, 1, 1, 0)
        self.fpn_c4 = C2f(c4 * 2, c4, n=max(1, int(3 * d)), shortcut=False)
        
        self.reduce_c4 = ConvBNSiLU(c4, c3, 1, 1, 0)
        self.fpn_c3 = C2f(c3 * 2, c3, n=max(1, int(3 * d)), shortcut=False)
        
        # PAN (bottom-up)
        self.down_c3 = ConvBNSiLU(c3, c3, 3, 2, 1)
        self.pan_c4 = C2f(c3 + c4, c4, n=max(1, int(3 * d)), shortcut=False)
        
        self.down_c4 = ConvBNSiLU(c4, c4, 3, 2, 1)
        self.pan_c5 = C2f(c4 + c5, c5, n=max(1, int(3 * d)), shortcut=False)
        
        self.out_channels = [c3, c4, c5]
    
    def forward(self, features):
        c3, c4, c5 = features
        
        # FPN
        p5 = self.reduce_c5(c5)
        p4 = self.fpn_c4(torch.cat([self.upsample(p5), c4], 1))
        
        p4_reduce = self.reduce_c4(p4)
        p3 = self.fpn_c3(torch.cat([self.upsample(p4_reduce), c3], 1))
        
        # PAN
        n3 = p3
        n4 = self.pan_c4(torch.cat([self.down_c3(n3), p4], 1))
        n5 = self.pan_c5(torch.cat([self.down_c4(n4), c5], 1))
        
        return n3, n4, n5


class YOLOv8Head(nn.Module):
    """YOLOv8检测头"""
    def __init__(self, in_channels, num_classes, reg_max=16):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        c3, c4, c5 = in_channels
        
        # 分类分支
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(c, c, 3, 1, 1),
                ConvBNSiLU(c, c, 3, 1, 1),
                nn.Conv2d(c, num_classes, 1)
            ) for c in [c3, c4, c5]
        ])
        
        # 回归分支
        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(c, c, 3, 1, 1),
                ConvBNSiLU(c, c, 3, 1, 1),
                nn.Conv2d(c, 4 * reg_max, 1)
            ) for c in [c3, c4, c5]
        ])
        
        # DFL层
        self.dfl = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.dfl.weight.data = torch.arange(reg_max).float().view(1, reg_max, 1, 1)
        self.dfl.weight.requires_grad = False
    
    def forward(self, features):
        outputs = []
        
        for i, (cls_conv, reg_conv, feat) in enumerate(
            zip(self.cls_convs, self.reg_convs, features)
        ):
            cls_out = cls_conv(feat)
            reg_out = reg_conv(feat)
            
            # 合并输出
            B, _, H, W = cls_out.shape
            cls_out = cls_out.view(B, self.num_classes, -1)
            reg_out = reg_out.view(B, 4, self.reg_max, -1)
            
            outputs.append({
                'cls': cls_out,
                'reg': reg_out,
                'stride': 2 ** (i + 3)
            })
        
        return outputs


class YOLOv8(nn.Module):
    """
    YOLOv8目标检测模型
    
    用于变电站设备缺陷检测
    """
    
    def __init__(self, num_classes: int = 80, scale: str = 'n', 
                 input_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        
        self.num_classes = num_classes
        self.scale = scale
        self.input_size = input_size
        
        # 网络结构
        self.backbone = YOLOv8Backbone(scale=scale)
        self.neck = YOLOv8Neck(self.backbone.out_channels, scale=scale)
        self.head = YOLOv8Head(self.neck.out_channels, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs


# =============================================================================
# U-Net模型 (语义分割)
# =============================================================================
class DoubleConv(nn.Module):
    """两个卷积层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net语义分割模型
    
    用于油位分割等任务
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 64, bilinear: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, num_classes, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


# =============================================================================
# CNN分类模型
# =============================================================================
class SimpleCNN(nn.Module):
    """
    简单CNN分类模型
    
    用于硅胶颜色分类、热成像异常分类等
    """
    
    def __init__(self, num_classes: int = 4, in_channels: int = 3,
                 input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1),
            SEBlock(64),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 256, 3, 2, 1),
            SEBlock(256),
            ConvBNReLU(256, 512, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18分类模型
    
    用于需要更强特征提取能力的分类任务
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [ResNetBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =============================================================================
# HRNet关键点检测模型
# =============================================================================
class HRNetBlock(nn.Module):
    """HRNet基本块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class HRNetSimple(nn.Module):
    """
    简化版HRNet关键点检测
    
    用于表计关键点检测
    """
    
    def __init__(self, num_keypoints: int = 8, in_channels: int = 3,
                 input_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # 初始卷积
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 高分辨率分支
        self.stage1 = nn.Sequential(
            HRNetBlock(64, 64),
            HRNetBlock(64, 64),
        )
        
        # 低分辨率分支
        self.transition = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.stage2_high = nn.Sequential(
            HRNetBlock(64, 64),
            HRNetBlock(64, 64),
        )
        
        self.stage2_low = nn.Sequential(
            HRNetBlock(128, 128),
            HRNetBlock(128, 128),
        )
        
        # 融合
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion = nn.Conv2d(64 + 128, 64, 1)
        
        # 关键点头
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, 1)
        )
    
    def forward(self, x):
        # 初始特征
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Stage 1
        x_high = self.stage1(x)
        
        # 过渡到低分辨率
        x_low = self.transition(x)
        
        # Stage 2
        x_high = self.stage2_high(x_high)
        x_low = self.stage2_low(x_low)
        
        # 融合
        x_low_up = self.upsample(x_low)
        x = torch.cat([x_high, x_low_up], dim=1)
        x = self.fusion(x)
        
        # 关键点热图
        heatmaps = self.head(x)
        
        return heatmaps


# =============================================================================
# CRNN OCR模型
# =============================================================================
class BidirectionalLSTM(nn.Module):
    """双向LSTM"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    """
    CRNN OCR识别模型
    
    用于表计数字识别、指示牌文字识别
    """
    
    def __init__(self, num_classes: int = 37, in_channels: int = 1,
                 hidden_size: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        
        # RNN序列建模
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"CNN输出高度应为1,当前为{h}"
        
        # 转换为序列
        conv = conv.squeeze(2)  # [b, c, w]
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        # RNN
        output = self.rnn(conv)
        
        return output


# =============================================================================
# RT-DETR模型 (简化版)
# =============================================================================
class RTDETR(nn.Module):
    """
    RT-DETR目标检测模型 (简化版)
    
    用于入侵检测等需要高精度的场景
    """
    
    def __init__(self, num_classes: int = 80, hidden_dim: int = 256,
                 num_queries: int = 300, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # 骨干网络 (使用ResNet18)
        self.backbone = ResNet18(num_classes=1000)  # 临时设置
        self.backbone.fc = nn.Identity()  # 移除分类头
        
        # 投影层
        self.input_proj = nn.Conv2d(512, hidden_dim, 1)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=1024
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=1024
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 预测头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, x):
        # 骨干特征
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # 投影
        x = self.input_proj(x)  # [B, hidden_dim, H, W]
        
        # 展平为序列
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # 编码
        memory = self.encoder(x)
        
        # 解码
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        hs = self.decoder(query, memory)  # [num_queries, B, C]
        
        # 预测
        hs = hs.permute(1, 0, 2)  # [B, num_queries, C]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }


# =============================================================================
# 模型工厂函数
# =============================================================================
def create_model(model_type: str, model_name: str, 
                 input_size: Tuple[int, int],
                 pretrained: bool = False,
                 plugin_name: str = None,
                 **kwargs) -> nn.Module:
    """
    创建模型实例
    
    Args:
        model_type: 模型类型 (detection, segmentation, classification, keypoint, ocr)
        model_name: 模型名称
        input_size: 输入尺寸
        pretrained: 是否使用预训练权重
        plugin_name: 插件名称
    
    Returns:
        模型实例
    """
    # 获取插件特定的类别数
    num_classes = kwargs.get('num_classes', _get_num_classes(plugin_name, model_type))
    
    if model_type == "detection":
        if "yolov8" in model_name.lower():
            scale = 'n'
            if 's' in model_name.lower():
                scale = 's'
            elif 'm' in model_name.lower():
                scale = 'm'
            elif 'l' in model_name.lower():
                scale = 'l'
            
            model = YOLOv8(num_classes=num_classes, scale=scale, input_size=input_size)
        elif "rtdetr" in model_name.lower():
            model = RTDETR(num_classes=num_classes)
        else:
            model = YOLOv8(num_classes=num_classes, scale='n', input_size=input_size)
    
    elif model_type == "segmentation":
        model = UNet(num_classes=num_classes)
    
    elif model_type == "classification":
        if "resnet" in model_name.lower():
            model = ResNet18(num_classes=num_classes)
        else:
            model = SimpleCNN(num_classes=num_classes, input_size=input_size)
    
    elif model_type == "keypoint":
        num_keypoints = kwargs.get('num_keypoints', 8)
        model = HRNetSimple(num_keypoints=num_keypoints, input_size=input_size)
    
    elif model_type == "ocr":
        charset_size = kwargs.get('charset_size', 37)  # 0-9 + a-z + blank
        model = CRNN(num_classes=charset_size)
    
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    logger.info(f"创建模型: {model_name} ({model_type}), 类别数: {num_classes}")
    
    return model


def _get_num_classes(plugin_name: str, model_type: str) -> int:
    """获取插件默认类别数"""
    CLASS_COUNTS = {
        "transformer": {"detection": 6, "classification": 4},
        "switch": {"detection": 8, "classification": 2},
        "busbar": {"detection": 8, "classification": 5},
        "capacitor": {"detection": 7, "classification": 3},
        "meter": {"detection": 5, "classification": 5},
    }
    
    default = 10
    if plugin_name in CLASS_COUNTS:
        return CLASS_COUNTS[plugin_name].get(model_type, default)
    return default


# =============================================================================
# 模型信息
# =============================================================================
def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / (1024 ** 2),  # 假设float32
    }
