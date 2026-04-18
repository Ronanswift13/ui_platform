# 00 Project Context

## 项目名称
输变电激光星芒监测平台 — 前端 UI 模块

## 项目类型
工业监测平台前端（Jinja2 + Vanilla JS + Bootstrap 5 + Chart.js）

## 技术栈
| 层级 | 技术 | 版本 |
|------|------|------|
| 模板引擎 | Jinja2 | - |
| CSS 框架 | Bootstrap | 5.3.2 |
| 图标 | Bootstrap Icons | 1.11.1 |
| 图表 | Chart.js | 4.4.1 |
| 3D 渲染 | Three.js | - |
| JavaScript | Vanilla ES6+ | - |
| 后端 | FastAPI | - |
| 通信 | WebSocket + REST | - |

## 插件体系
平台集成 16 个检测插件，分为两大类：

### 室外监测（10 个）
1. transformer_inspection — 主变自主巡视
2. switch_inspection — 开关间隔巡视
3. busbar_inspection — 母线自主巡视
4. capacitor_inspection — 电容器巡视
5. bird_monitoring — 鸟类监控
6. acoustic_monitoring — 声学监测
7. gas_detection — 气体泄漏检测
8. hyperspectral_detection — 高光谱缺陷检测
9. slam_mapping — SLAM 三维建图
10. multimodal_fusion — 多模态融合诊断

### 室内监测（6 个）
1. indoor_fence — 室内电子围栏
2. animal_detection — 动物入侵检测
3. temperature_monitoring — 温度监测
4. device_monitoring — 设备状态监测
5. fire_detection — 消防监测
6. meter_reading — 表计读数

## 核心页面
| 路由 | 页面 | 说明 |
|------|------|------|
| `/` `/cockpit` | cockpit.html | 主控驾驶舱（V4.0 新增） |
| `/home` | index.html | 原始首页（保留） |
| `/outdoor` | outdoor_center_v4.html | 室外监测中心 |
| `/indoor` | indoor_center.html | 室内监测中心 |
| `/dashboard` | dashboard.html | V2.0 监测中心 |
| `/unified-dashboard` | unified_dashboard.html | V3.0 统一仪表盘 |
| `/plugins` | plugin_manager.html | 插件管理 |
| `/training` | training.html | 模型训练 |
| `/data-import` | data_import.html | 数据导入 |
| `/settings` | settings.html | 系统设置 |
| `/replay` | replay.html | 回放 |

## API 端点
- `/api/plugins/list` — 获取所有插件列表
- `/api/plugins/enabled` — 获取已启用插件
- `/api/plugins/enable/{id}` — 启用插件
- `/api/plugins/disable/{id}` — 禁用插件
- `/api/plugins/{id}/info` — 插件详情
- `/api/tasks/run` — 运行检测任务
- `/api/outdoor/*` — 室外监测 API
- `/api/indoor/*` — 室内监测 API

## 设计参考
主控驾驶舱参考云南电网生产运行支持系统设计：
- 深蓝色大屏主题
- 左侧分类导航 + 中央多面板仪表盘 + 右侧实时数据
- 环形图、柱状图、折线图等数据可视化
- 实时告警列表与图片展示
