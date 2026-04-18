"""
训练流水线模块

五阶段流水线:
1. ingestion   - 数据摄入与 manifest 校验
2. preprocessing - 数据预处理 (按 task_type 分流)
3. training    - 模型训练 (按 task_type 分流)
4. evaluation  - 模型评估
5. export      - 模型导出 (ONNX / TensorRT)
"""
