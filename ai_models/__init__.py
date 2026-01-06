"""
AI模型开发与部署模块
输变电站全自动AI巡检方案

模块结构:
- training: 模型训练模块 (SLAM、声学、时序、融合)
- deployment: 模型部署模块 (ONNX/TensorRT转换)
- research: 研究模块 (图优化SLAM、小样本学习、不确定性量化等)

版本: 1.0.0
"""

from pathlib import Path

# 模块版本
__version__ = "1.0.0"

# 模块根目录
AI_MODELS_ROOT = Path(__file__).parent

# 导出子模块
__all__ = [
    "training",
    "deployment",
    "research",
    "AI_MODELS_ROOT",
    "__version__"
]
