# 06_refactor_policy

## 1. 当前重构原则

1. 先补最小治理，再做算法升级。
2. 先锁住 `process()` 合同，再修内部实现。
3. 先对齐配置与元数据，再引入真实模型。
4. 先修光谱维度判断，再谈复杂谱分析。

## 2. 允许的低风险重构

1. 补齐 `.agent_skills/00~08`
2. 新增 `scripts/run_sanity_checks.sh`
3. 新增 `tests/test_plugin_process.py`
4. 新增 `configs/default.yaml`
5. 对齐 `_parse_config()` 与 manifest 默认配置
6. 修正文案里缺失的 `run_standalone.py` 引用

## 3. 中风险重构（建议先补测试）

1. 修正 3D 输入的 band 轴推断
2. 让 `analysis_type` 真正控制输出分支
3. 让 `confidence_threshold` / `pca_components` 真正参与计算
4. 打通 `_model_registry` 的真实使用路径

## 4. 高风险重构（需人工确认）

1. 修改 `process()` 输入输出合同
2. 引入真实高光谱模型、PCA、分割或定位算法
3. 修改 standalone 服务暴露方式
4. 引入真实文件读取、相机接入或数据库写入
5. 增加外部网络访问

## 5. 当前强制流程

1. 先读 `00/01/02/03`
2. 明确改动属于：
   - 文档治理
   - 测试补齐
   - 配置修复
   - 算法/行为修复
3. 若改 `plugin.py` / `manifest.json` / `requirements.txt`
   - 至少跑 demo 回放
   - 至少跑一次基础 `init + process` 校验
4. 若修 band 轴逻辑
   - 必须补形状回归用例或 sanity 检查
