"""
Schema校验器

提供统一的数据验证功能:
- 验证插件输出格式
- 验证配置文件格式
- 验证任务参数
"""


from __future__ import annotations
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from platform_core.exceptions import SchemaValidationError
from platform_core.schema.models import PluginOutput, RecognitionResult

T = TypeVar("T", bound=BaseModel)


class SchemaValidator:
    """Schema校验器"""

    @staticmethod
    def validate(data: dict[str, Any], schema: Type[T], strict: bool = True) -> T:
        """
        验证数据是否符合Schema

        Args:
            data: 待验证的数据字典
            schema: Pydantic模型类
            strict: 是否严格模式 (不允许多余字段)

        Returns:
            验证通过的模型实例

        Raises:
            SchemaValidationError: 验证失败时抛出
        """
        try:
            if strict:
                return schema.model_validate(data, strict=True)
            return schema.model_validate(data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                errors.append({
                    "loc": ".".join(str(x) for x in error["loc"]),
                    "msg": error["msg"],
                    "type": error["type"],
                })
            raise SchemaValidationError(schema.__name__, errors) from e

    @staticmethod
    def validate_partial(data: dict[str, Any], schema: Type[T]) -> dict[str, Any]:
        """
        部分验证 - 只验证提供的字段

        用于更新操作时只验证部分字段
        """
        # 获取schema中定义的字段
        schema_fields = set(schema.model_fields.keys())
        # 只保留schema中存在的字段
        filtered_data = {k: v for k, v in data.items() if k in schema_fields}

        try:
            # 创建临时实例进行验证
            for field_name, field_value in filtered_data.items():
                field_info = schema.model_fields.get(field_name)
                if field_info and field_info.annotation:
                    # 使用Pydantic进行类型验证
                    schema.model_validate({**{f: None for f in schema.model_fields if f != field_name}, field_name: field_value})
        except ValidationError as e:
            errors = [{"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"], "type": err["type"]} for err in e.errors()]
            raise SchemaValidationError(f"{schema.__name__}[partial]", errors) from e

        return filtered_data

    @staticmethod
    def is_valid(data: dict[str, Any], schema: Type[T]) -> bool:
        """检查数据是否有效 (不抛出异常)"""
        try:
            schema.model_validate(data)
            return True
        except ValidationError:
            return False


def validate_plugin_output(output: dict[str, Any], plugin_id: str, strict: bool = True) -> PluginOutput:
    """
    验证插件输出格式

    Args:
        output: 插件返回的原始数据
        plugin_id: 插件ID (用于错误信息)
        strict: 是否严格模式

    Returns:
        验证通过的PluginOutput实例

    Raises:
        SchemaValidationError: 输出格式不符合规范
    """
    try:
        validated = PluginOutput.model_validate(output)

        # 额外验证每个结果
        if strict:
            for i, result in enumerate(validated.results):
                # 确保必要字段存在
                if not result.task_id:
                    raise SchemaValidationError(
                        f"PluginOutput.results[{i}]",
                        [{"loc": "task_id", "msg": "task_id不能为空", "type": "value_error"}]
                    )
                if result.confidence < 0 or result.confidence > 1:
                    raise SchemaValidationError(
                        f"PluginOutput.results[{i}]",
                        [{"loc": "confidence", "msg": "置信度必须在0-1之间", "type": "value_error"}]
                    )

        return validated

    except ValidationError as e:
        errors = [
            {
                "loc": ".".join(str(x) for x in err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
            }
            for err in e.errors()
        ]
        raise SchemaValidationError(f"PluginOutput[{plugin_id}]", errors) from e


def validate_recognition_result(result: dict[str, Any]) -> RecognitionResult:
    """验证单个识别结果"""
    return SchemaValidator.validate(result, RecognitionResult)
