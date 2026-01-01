# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练模块
================================

训练模块统一管理所有模型训练相关功能。

目录结构:
    training/
    ├── __init__.py                # 本文件
    ├── train_main.py              # 主训练脚本
    ├── train_mac.sh               # Mac训练脚本
    ├── prepare_training_data.py   # 数据准备脚本
    ├── evaluate_training.py       # 训练评估脚本
    ├── configs/                   # 训练配置
    │   └── training_config.yaml   # 训练参数配置
    ├── checkpoints/               # 模型检查点
    │   ├── transformer/           # A组 - 主变巡视
    │   ├── switch/                # B组 - 开关间隔
    │   ├── busbar/                # C组 - 母线巡视
    │   ├── capacitor/             # D组 - 电容器
    │   └── meter/                 # E组 - 表计读数
    ├── exports/                   # ONNX临时导出
    ├── logs/                      # 训练日志
    ├── data/                      # 训练数据
    └── results/                   # 训练结果

使用方法:
    # 从项目根目录运行
    python train.py --mode demo
    python train.py --mode plugin --plugin transformer --epochs 30

    # 或直接运行训练模块
    python -m training.train_main --mode demo
"""

__version__ = "2.0.0"
__author__ = "破夜绘明团队"
