# Runtime Truth / Quality Gate Sequence

给后续插件代理复用的最小顺序约束：

1. 先做 Runtime Truth 收口，再谈精度。
2. 先冻结 Runtime Supported Labels，再谈扩类。
3. 质量门禁先做 Tri-state（`pass/soft_fail/hard_fail`），再做 Real Model Validation。
4. Simulator 必须校准真实 `check_quality_gate()`，不得只做视觉演示。
5. Real DL 接入要放在 P0/P1/P2 之后进行，且不得回退 runtime truth 字段。
6. 训练、样本扩类与精度叙事，必须放在模型接入和质量门禁稳定之后。

停止条件：

- 如果 tri-state 需要改 SDK 签名，停止。
- 如果 simulator 不能稳定复现实门禁结果，先修校准，不要推进 real_dl。
- 如果 blocked 类需要靠重命名才能“看起来支持”，停止。
