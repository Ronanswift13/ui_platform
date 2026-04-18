# 02 UI 契约

## 主题规范
- 深蓝色大屏主题（CSS 变量 `--ck-*` 体系）
- 室内/室外导航入口必须保留

## 三态覆盖要求
所有数据面板必须覆盖:
1. **loading** — 加载中状态
2. **empty** — 无数据状态
3. **error** — 错误状态

## 组件边界
- 展示组件不直接发起请求
- 数据获取统一通过 API 层

## 验证命令

```bash
./scripts/check_ui_contract.sh
./scripts/check_three_state_coverage.sh
```
