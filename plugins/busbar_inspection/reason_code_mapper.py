"""原因码映射模块

统一内部原因码到外部原因码的映射。
遵循 02_algorithm_contract.md 第 2.4 节的原因码统一契约。
"""

from typing import Dict, Optional


class ReasonCodeMapper:
    """原因码映射器"""

    # 内部码到外部码的映射
    INTERNAL_TO_EXTERNAL = {
        1001: 103,  # 模糊/失焦
        1002: 101,  # 过曝
        1003: 101,  # 低对比
        1004: 102,  # 遮挡/不可见
        1005: 101,  # 欠曝
        2001: 201,  # 目标过小，需要变焦
        2002: 202,  # 检测不稳定
        3001: 301,  # 环境干扰（鸟类）
        3002: 301,  # 环境干扰（飞虫）
        3003: 301,  # 环境干扰（水滴）
    }

    # 外部原因码描述
    EXTERNAL_DESCRIPTIONS = {
        101: "光照异常（过曝/欠曝/低对比）",
        102: "遮挡/不可见",
        103: "模糊/失焦",
        104: "雨雾/低能见度",
        105: "运动干扰",
        201: "目标过小，需要变焦",
        202: "检测不稳定",
        301: "环境干扰（鸟类/飞虫/水滴）",
    }

    @classmethod
    def map_to_external(cls, internal_code: Optional[int]) -> Optional[int]:
        """将内部原因码映射到外部原因码

        Args:
            internal_code: 内部原因码

        Returns:
            外部原因码，如果映射不存在则返回 None
        """
        if internal_code is None:
            return None
        return cls.INTERNAL_TO_EXTERNAL.get(internal_code)

    @classmethod
    def get_description(cls, external_code: Optional[int]) -> str:
        """获取外部原因码的描述

        Args:
            external_code: 外部原因码

        Returns:
            原因码描述，如果不存在则返回"未知原因"
        """
        if external_code is None:
            return "未知原因"
        return cls.EXTERNAL_DESCRIPTIONS.get(external_code, "未知原因")

    @classmethod
    def map_and_describe(cls, internal_code: Optional[int]) -> tuple[Optional[int], str]:
        """映射内部码并返回外部码和描述

        Args:
            internal_code: 内部原因码

        Returns:
            (外部原因码, 描述) 元组
        """
        external_code = cls.map_to_external(internal_code)
        description = cls.get_description(external_code)
        return external_code, description
