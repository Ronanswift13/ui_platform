# -*- coding: utf-8 -*-
"""
基础插件抽象类
所有插件都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """
    插件基类
    定义插件的标准接口
    """
    
    # 子类必须覆盖这些属性
    PLUGIN_ID: str = 'base_plugin'
    PLUGIN_NAME: str = '基础插件'
    VERSION: str = '1.0.0'
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化插件
        
        Args:
            config: 插件配置
        """
        self.config = config or {}
        self.model_registry = None
        self.initialized = False
        self._enabled = True
        
    @abstractmethod
    def init(self, context: Optional[Dict] = None) -> bool:
        """
        初始化插件资源
        
        Args:
            context: 上下文信息，可能包含其他插件引用、全局配置等
            
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            处理结果字典，必须包含 'success' 字段
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """
        关闭插件，释放资源
        """
        pass
    
    def set_model_registry(self, registry):
        """
        设置模型注册器
        
        Args:
            registry: ModelRegistry实例
        """
        self.model_registry = registry
        logger.info(f"[{self.PLUGIN_ID}] 模型注册器已设置")
    
    def get_plugin_info(self) -> Dict:
        """
        获取插件信息
        
        Returns:
            插件信息字典
        """
        return {
            'plugin_id': self.PLUGIN_ID,
            'plugin_name': self.PLUGIN_NAME,
            'version': self.VERSION,
            'initialized': self.initialized,
            'enabled': self._enabled,
            'has_model_registry': self.model_registry is not None
        }
    
    def enable(self):
        """启用插件"""
        self._enabled = True
        logger.info(f"[{self.PLUGIN_ID}] 插件已启用")
    
    def disable(self):
        """禁用插件"""
        self._enabled = False
        logger.info(f"[{self.PLUGIN_ID}] 插件已禁用")
    
    @property
    def is_enabled(self) -> bool:
        """检查插件是否启用"""
        return self._enabled and self.initialized
    
    def validate_input(self, input_data: Dict, required_fields: List[str]) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            required_fields: 必需字段列表
            
        Returns:
            是否验证通过
        """
        for field in required_fields:
            if field not in input_data:
                logger.warning(f"[{self.PLUGIN_ID}] 缺少必需字段: {field}")
                return False
        return True
    
    def safe_process(self, input_data: Dict) -> Dict:
        """
        安全处理包装器
        自动处理异常
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        if not self.is_enabled:
            return {
                'success': False,
                'error': f'插件 {self.PLUGIN_ID} 未启用或未初始化'
            }
        
        try:
            return self.process(input_data)
        except Exception as e:
            logger.error(f"[{self.PLUGIN_ID}] 处理异常: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
