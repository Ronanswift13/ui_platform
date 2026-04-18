"""配置映射适配器

将嵌套的 YAML 配置映射到统一的配置视图，供算法层使用。
遵循 02_algorithm_contract.md 第 2.6 节的配置映射契约。
"""

from typing import Any, Dict, Optional
import yaml


class ConfigAdapter:
    """配置适配器，提供统一的配置访问接口"""

    # 配置映射表：统一键 -> (YAML路径, 默认值)
    CONFIG_MAPPING = {
        'confidence_threshold': ('thresholds.conf_thr', 0.25),
        'nms_threshold': ('thresholds.nms_iou', 0.50),
        'tile_size': ('tiling.tile_size', 1280),
        'tile_overlap': ('tiling.overlap', 320),
        'clarity_threshold': ('quality.blur_thr', 0.35),
        'min_object_size_px': ('zoom.min_obj_px', 18),
        'target_size_px': ('zoom.target_px', 90),
        'zoom_min': ('zoom.zmin', 1.0),
        'zoom_max': ('zoom.zmax', 12.0),
    }

    def __init__(self, config_dict: Dict[str, Any]):
        """初始化配置适配器

        Args:
            config_dict: 从 YAML 加载的配置字典
        """
        self._config = config_dict
        self._cache: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取配置值

        Args:
            key: 统一配置键
            default: 默认值（如果提供，覆盖映射表中的默认值）

        Returns:
            配置值

        Raises:
            KeyError: 如果键不在映射表中
        """
        if key in self._cache:
            return self._cache[key]

        if key not in self.CONFIG_MAPPING:
            raise KeyError(f"Unknown config key: {key}")

        yaml_path, default_value = self.CONFIG_MAPPING[key]
        if default is not None:
            default_value = default

        value = self._get_nested_value(yaml_path, default_value)
        self._cache[key] = value
        return value

    def _get_nested_value(self, path: str, default: Any) -> Any:
        """从嵌套字典中获取值

        Args:
            path: 点分隔的路径，如 'thresholds.conf_thr'
            default: 默认值

        Returns:
            配置值或默认值
        """
        keys = path.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ConfigAdapter':
        """从 YAML 文件创建配置适配器

        Args:
            yaml_path: YAML 文件路径

        Returns:
            ConfigAdapter 实例
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
