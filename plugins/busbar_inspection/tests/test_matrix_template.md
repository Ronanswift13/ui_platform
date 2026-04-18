# Busbar Inspection Test Matrix

| Area | Scenario | Type | Command / Fixture | Expected Result | Owner |
|------|----------|------|-------------------|-----------------|-------|
| plugin | 空 ROI 输入 | unit | `tests/test_standalone.py` | 返回空列表，不抛异常 | C组 |
| plugin | quality_failed 告警映射 | unit | `tests/test_plugin_postprocess.py` | 输出 WARNING 且含建议动作 | C组 |
| detector | 模糊图像门禁 | unit | `tests/test_quality_gate_contract.py` | 原因码映射到 103 | C组 |
| detector | 过曝/低对比门禁 | unit | `tests/test_quality_gate_contract.py` | 原因码映射到 101 | C组 |
| detector | 小目标变焦建议 | unit | `tests/test_detector_contract.py` | `suggested_action` 非 NONE | C组 |
| detector | 切片 remap + NMS | unit | `tests/test_bbox_contract.py` | bbox 不越界，重复框下降 | C组 |
| contract | 配置路径映射 | unit | `tests/test_config_contract.py` | YAML 变更被算法层读取 | C组 |
| contract | 原因码统一映射 | unit | `tests/test_reason_code_contract.py` | 内外部码映射一致 | C组 |
| integration | infer -> postprocess | integration | `tests/test_plugin_contract.py` | 输出字段完整，告警等级正确 | C组 |
| regression | 回放样本稳定性 | regression | `tests/regression/` | 关键指标不劣化 | C组 |
