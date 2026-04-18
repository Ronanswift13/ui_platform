# Multimodal Fusion Upload Examples

本目录提供 `multimodal_fusion` 的上传样例。

关键点：

- `manifest.json` 描述数据集级约束
- `samples.jsonl` 描述单样本对齐
- 每个样本只声明自己真实可用的模态
- 缺失模态不需要补空文件

推荐先从 `aligned_sparse_example/` 开始联调。
