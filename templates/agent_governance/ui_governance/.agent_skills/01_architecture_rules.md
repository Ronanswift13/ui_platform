# 01 架构规则 — UI

## 架构不变量

1. 所有数据面板必须显式处理 loading / empty / error 三态。
2. 插件数据统一通过 API 获取，前端负责聚合展示。
3. 图表使用已引入的库，不得新增大型图表依赖。
4. CSS 布局优先 CSS Grid，避免绝对定位。
5. 页面文件超过 400 行必须拆分为组件。
6. 展示组件中不直接发起数据请求。

## 反模式检查

```bash
./scripts/check_ui_contract.sh
./scripts/check_three_state_coverage.sh
```
