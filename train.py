#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练入口脚本
====================================

所有训练相关文件统一位于 training/ 目录下。
本脚本是根目录的便捷入口。

使用方法:
---------
# 演示模式 (快速测试)
python train.py --mode demo

# 训练主变巡视 (A组)
python train.py --mode plugin --plugin transformer --epochs 30

# 训练开关间隔 (B组)
python train.py --mode plugin --plugin switch --epochs 30

# 训练母线巡视 (C组)
python train.py --mode plugin --plugin busbar --epochs 30

# 训练电容器 (D组)
python train.py --mode plugin --plugin capacitor --epochs 30

# 训练表计读数 (E组)
python train.py --mode plugin --plugin meter --epochs 30

# 训练所有模型
python train.py --mode all --epochs 50

# 查看可训练的模型信息
python train.py --mode info

训练输出目录:
-------------
所有训练产物统一保存在 training/ 目录下:
  - training/checkpoints/  检查点文件 (.pth)
  - training/exports/      ONNX临时导出
  - training/logs/         训练日志
  - training/results/      训练结果摘要

部署模型:
---------
训练完成后，ONNX模型会自动复制到:
  - models/{plugin_name}/  生产部署目录

作者: 破夜绘明团队
"""

import sys
import os

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入并运行训练主模块
if __name__ == "__main__":
    # 直接执行 training/train_main.py
    train_main_path = os.path.join(PROJECT_ROOT, "training", "train_main.py")

    if os.path.exists(train_main_path):
        # 使用 exec 执行训练脚本，保持参数传递
        with open(train_main_path, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(compile(code, train_main_path, 'exec'), {'__name__': '__main__', '__file__': train_main_path})
    else:
        print(f"错误: 找不到训练脚本 {train_main_path}")
        print("请确保 training/train_main.py 存在")
        sys.exit(1)
