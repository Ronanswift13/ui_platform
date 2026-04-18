# Regression Tests（busbar_inspection）

用于沉淀历史缺陷回归与关键指标保护。

## 最低要求
1. 每个线上缺陷修复必须新增 1 条回归测试。
2. 至少覆盖：质量门禁原因码、配置映射、bbox 越界防护。
3. 回归测试应尽量固定输入，不依赖随机数据。

## 建议命名
- `test_regression_reason_code_*.py`
- `test_regression_config_mapping_*.py`
- `test_regression_bbox_remap_*.py`
