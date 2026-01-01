#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pylance 类型错误修复工具
========================

修复 extended_inference_engine.py 中的类型检查警告
主要解决 .squeeze() 方法在 Union 类型上的访问问题

使用方法:
    python fix_type_errors.py <path_to_extended_inference_engine.py>
"""

import re
import sys
from pathlib import Path


def fix_squeeze_type_errors(content: str) -> str:
    """
    修复 .squeeze() 调用的类型错误
    
    将:
        outputs[0].squeeze()
    改为:
        np.asarray(outputs[0]).squeeze()
    
    或添加类型检查
    """
    
    # 模式1: 修复 outputs[N].squeeze() 调用
    pattern1 = r'(outputs\[\d+\])\.squeeze\(\)'
    replacement1 = r'np.asarray(\1).squeeze()'
    content = re.sub(pattern1, replacement1, content)
    
    # 模式2: 修复 result.xxx = outputs[N].squeeze() 
    # 已经在上面的模式中处理
    
    return content


def add_type_ignore_comments(content: str) -> str:
    """
    在特定行添加 # type: ignore 注释
    """
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        # 如果行包含 .squeeze() 调用且没有 type: ignore
        if '.squeeze()' in line and '# type: ignore' not in line:
            # 检查是否在 outputs 或类似变量上调用
            if 'outputs[' in line or 'result.' in line:
                line = line.rstrip() + '  # type: ignore[union-attr]'
        
        modified_lines.append(line)
    
    return '\n'.join(modified_lines)


def fix_session_run_return_type(content: str) -> str:
    """
    修复 session.run() 返回类型问题
    
    在调用 session.run 后添加类型断言
    """
    # 添加辅助函数定义（如果不存在）
    helper_function = '''
def _ensure_ndarray(value: Any) -> np.ndarray:
    """确保值为 numpy 数组"""
    if isinstance(value, np.ndarray):
        return value
    elif hasattr(value, 'numpy'):
        return value.numpy()
    else:
        return np.asarray(value)

'''
    
    # 检查是否已存在该函数
    if '_ensure_ndarray' not in content:
        # 在类定义之前插入辅助函数
        # 找到第一个 class 定义
        class_match = re.search(r'^class\s+\w+', content, re.MULTILINE)
        if class_match:
            insert_pos = class_match.start()
            content = content[:insert_pos] + helper_function + content[insert_pos:]
    
    return content


def create_type_stubs() -> str:
    """
    创建类型存根文件内容
    """
    return '''# extended_inference_engine.pyi
# 类型存根文件 - 解决 Pylance 警告

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# 定义 ONNX 推理输出类型
ONNXOutput = NDArray[np.floating[Any]]
ONNXOutputs = List[ONNXOutput]

class ExtendedModelConfig:
    model_id: str
    model_path: str
    model_type: str
    ...

class ExtendedInferenceResult:
    success: bool
    model_id: str
    inference_time_ms: float
    error_message: Optional[str]
    outputs: Dict[str, np.ndarray]
    ...
'''


def generate_fixed_code_template() -> str:
    """
    生成修复后的代码模板
    展示如何正确处理类型
    """
    return '''
# ============================================================================
# 修复后的代码模式示例
# ============================================================================

# 方法1: 使用 np.asarray 包装
def _do_infer_fixed_v1(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
    """修复版本1 - 使用 np.asarray"""
    
    # ... 前置代码 ...
    
    if self._session is None:
        return ExtendedInferenceResult(
            success=False,
            model_id=self.config.model_id,
            inference_time_ms=0,
            error_message="ONNX会话未初始化"
        )
    
    # 执行推理
    outputs = self._session.run(self._output_names, input_data)
    
    # 修复: 确保是 ndarray 后再调用 squeeze
    if len(outputs) >= 1:
        output_array = np.asarray(outputs[0])
        result.predictions = output_array.squeeze()
    
    return result


# 方法2: 添加类型检查
def _do_infer_fixed_v2(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
    """修复版本2 - 添加类型检查"""
    
    # ... 前置代码 ...
    
    outputs = self._session.run(self._output_names, input_data)
    
    # 修复: 显式类型检查
    if len(outputs) >= 1:
        output = outputs[0]
        if isinstance(output, np.ndarray):
            result.predictions = output.squeeze()
        else:
            result.predictions = np.asarray(output).squeeze()
    
    return result


# 方法3: 使用类型忽略（最简单但不推荐长期使用）
def _do_infer_fixed_v3(self, inputs: Dict[str, Any]) -> ExtendedInferenceResult:
    """修复版本3 - 类型忽略注释"""
    
    outputs = self._session.run(self._output_names, input_data)
    
    if len(outputs) >= 1:
        result.predictions = outputs[0].squeeze()  # type: ignore[union-attr]
    
    return result
'''


def main():
    """主函数"""
    print("=" * 60)
    print("Pylance 类型错误修复工具")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.exists():
            print(f"\n处理文件: {file_path}")
            
            # 读取文件
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # 应用修复
            content = fix_squeeze_type_errors(content)
            content = fix_session_run_return_type(content)
            
            # 备份原文件
            backup_path = file_path.with_suffix('.py.bak')
            backup_path.write_text(original_content, encoding='utf-8')
            print(f"原文件已备份到: {backup_path}")
            
            # 写入修复后的文件
            file_path.write_text(content, encoding='utf-8')
            print(f"修复完成: {file_path}")
            
            # 生成类型存根
            stub_path = file_path.with_suffix('.pyi')
            stub_path.write_text(create_type_stubs(), encoding='utf-8')
            print(f"类型存根已生成: {stub_path}")
            
        else:
            print(f"错误: 文件不存在 - {file_path}")
    else:
        print("\n使用方法:")
        print("  python fix_type_errors.py <path_to_file.py>")
        print("\n或者，你可以手动应用以下修复模式:")
        print(generate_fixed_code_template())
    
    print("\n" + "=" * 60)
    print("建议: 这些 Pylance 警告不影响运行时功能")
    print("可以在 VS Code 设置中调整 Pylance 严格程度:")
    print('  "python.analysis.typeCheckingMode": "basic"')
    print("=" * 60)


if __name__ == "__main__":
    main()
