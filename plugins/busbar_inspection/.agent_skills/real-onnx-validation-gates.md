# Real ONNX Validation Gates

给后续插件代理复用的 P3 最小顺序约束：

1. 先完成 runtime truth，再做 real_dl 验证。
2. 先冻结 runtime supported labels，再检查模型类别兼容性。
3. 先完成 tri-state quality gate，再进入 ONNX preflight。
4. real_dl 验证必须先过 preflight：`model exists -> manifest/class_map -> label contract -> session -> output shape`。
5. `model file exists != real_dl ready`。
6. `session ready != precision improved`。
7. 接入失败时必须保持 `traditional_fallback`，不得回到 simulation 伪 ready。

停止条件：

- 没有兼容当前标签契约的 ONNX 资产时，停止并保留 fallback。
- manifest / class_map 缺失且无法安全补齐时，停止。
- output tensor 与现有 detector 后处理明显不兼容时，停止。
- 如果要靠扩标签、改 SDK 或训练升级才能接入，停止。
