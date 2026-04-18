# 02_algorithm_contract

本插件当前更接近“点云处理/地图维护算法合同 + 若干本地服务接口”，因此沿用 `02_algorithm_contract` 命名。

## 1. 当前平台与业务入口契约

### 1.1 SDK 兼容入口

当前存在但基本为空壳：

- `infer(frame, rois, context) -> []`
- `postprocess(results, rules) -> []`
- `healthcheck() -> HealthStatus`

说明：

1. 这些方法满足部分 SDK 兼容性。
2. 当前真实业务不走 `infer()`。

### 1.2 真实业务入口

当前主入口是：

```python
process_point_cloud(points: np.ndarray, sensor_pose: Optional[Pose] = None) -> Dict[str, Any]
```

补充说明：

1. `sensor_pose` 当前签名存在，但主逻辑没有真实使用它。
2. 当前没有通用 `process()`。

## 2. 输入契约

### 2.1 `create_standalone(config=None)`

当前真实行为：

1. 优先读取 `manifest.json`
2. 若 `config is None`，尝试加载 `configs/default.yaml`
3. 但当前插件目录没有 `configs/default.yaml`
4. SDK `load_plugin_config()` 缺文件时返回空字典，因此最终回到默认配置

### 2.2 `init(model_registry=None)`

当前真实签名是 `model_registry`，不是 `config`。

这意味着：

1. 传入真正的 registry 对象时，会调用 `set_model_registry()` 并置 `dl_enabled=True`
2. 传入非空 dict 配置时，也会被误当成 registry
3. 当前不会把 dict 配置应用到 `point_processor` / `occupancy_map` / `icp_matcher`

### 2.3 `process_point_cloud()` 输入

```python
points: np.ndarray               # Nx3 或 Nx4，至少前三列为 xyz
sensor_pose: Optional[Pose]      # 当前基本未使用
```

当前真实行为：

1. 不要求先 `init()`，未初始化也能处理。
2. 空数组或预处理后无有效点云时返回失败。
3. `points` 中的 `NaN` / `inf` 会在预处理阶段被过滤。

## 3. 输出契约

### 3.1 `process_point_cloud()` 成功返回（当前真实结构）

常见字段：

```python
{
    "success": True,
    "frame_id": int,
    "timestamp": str,
    "processed_points": int,
    "ground_points": int,
    "object_points": int,
    "detected_objects": int,
    "objects": list[dict],
    "pose": {
        "x": float,
        "y": float,
        "z": float,
        "roll": float,
        "pitch": float,
        "yaw": float,
    },
    # 可选
    "registration_error": float,
    "subsidence_alerts": list[dict],
}
```

说明：

1. `registration_error` 只在已有 `last_keyframe_points` 时出现。
2. `subsidence_alerts` 只在存在监测点且触发阈值时出现。

### 3.2 失败返回

```python
{
    "success": False,
    "frame_id": int,
    "timestamp": str,
    "error": str,
}
```

已确认失败场景：

1. 预处理后点云为空 -> `error = "有效点云为空"`
2. 算法异常 -> `error = str(e)`

### 3.3 其他本地服务接口

当前还暴露：

- `register_device(device_id, location, device_type, metadata=None)`
- `locate_device(device_id)`
- `find_nearest_devices(location, radius=10.0, device_type=None)`
- `plan_inspection_path(start, device_ids)`
- `add_subsidence_monitor(point_id, location, initial_height=None)`
- `get_map_data(format='2d'|'3d')`
- `detect_obstacles(robot_pose, detection_range=5.0)`
- `export_map(filepath, format='json')`
- `get_status()`

这些是本地服务接口，不等于统一平台标准合同。

## 4. 配置与依赖契约

### 4.1 当前真实配置来源

| 配置项 | 当前来源 | 备注 |
|---|---|---|
| 点云体素大小 | `_default_config()['voxel_size']` | 默认 0.1 |
| 地面阈值 | `_default_config()['ground_threshold']` | 默认 0.3 |
| 聚类参数 | `_default_config()` | 当前真实生效 |
| 地图分辨率/原点/尺寸 | `_default_config()` | 当前真实生效 |
| ICP 参数 | `_default_config()` | 当前真实生效 |
| 沉降阈值 | `_default_config()` | 当前真实生效 |

### 4.2 manifest 与实现关系

`manifest.json.default_config` 当前只声明：

- `voxel_size`
- `max_range`
- `min_range`

而 `plugin.py._default_config()` 真实包含更多参数。

因此当前真实情况是：

1. manifest 只覆盖了默认配置的很小一部分。
2. 又因为 `init()` 不接收配置 dict，manifest/YAML 配置当前并不能稳定注入运行组件。

### 4.3 外部依赖 / 模型依赖

1. `requirements.txt` 当前依赖：
   - `darkbreaker-sdk`
   - `numpy`
   - `pydantic`
   - `pyyaml`
2. `manifest.json` 还声明了 `scipy`
3. 当前 `plugin.py` 主链路没有真实深度学习模型调用
4. `set_model_registry()` 只会：
   - 保存 registry
   - 将 `dl_enabled=True`
5. 当前 `process_point_cloud()` 并不会使用 model registry

### 4.4 并行实现依赖

`semantic_slam_plugin.py` 自带语义分割/变化检测相关能力和模型路径参数，但当前 manifest 并未接它，因此只能写成“候选演进方向”，不能写成现状。

## 5. 降级策略

| 场景 | 当前真实行为 |
|---|---|
| `configs/default.yaml` 缺失 | `create_standalone()` 仍可成功，回到 `_default_config()` |
| 未设置 model registry | 纯规则/几何算法链路照常运行 |
| 传入空点云或无效点云 | 返回失败，不更新地图 |
| 无关键帧 | 跳过 ICP 配准，不产出 `registration_error` |
| 无沉降监测点 | 跳过沉降告警 |
| 路径不可达 | `plan_inspection_path()` 返回失败 |
| 未 `init()` | 当前仍可处理点云，这是实现事实，不代表设计正确 |
| `shutdown()` 后 | 当前仍可能 `healthcheck()==OK`，这是实现缺陷，不应当作正式降级设计 |

## 6. 已验证的最小事实链路

1. `Plugin.create_standalone()` 可用
2. `process_point_cloud(np.random.randn(100, 3))` 返回成功
3. `demo/run_demo.py` 可跑通
4. `tests/test_standalone.py` 本地通过
5. 未 `init()` 时 `process_point_cloud()` 仍可运行
6. `shutdown()` 后 `healthcheck()` 仍返回 `healthy=True`

因此当前合同应写成：

- 点云处理与地图更新：基础可用
- 平台标准数据入口：不完整，依赖 `process_point_cloud()` 这条本地业务接口
- 配置注入：当前存在明显语义问题
- 深度学习：仅有开关/占位，不是当前主链路能力
