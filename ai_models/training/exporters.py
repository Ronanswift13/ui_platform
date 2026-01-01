#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型导出与验证模块
=====================

支持:
- PyTorch模型导出为ONNX
- ONNX模型验证
- ONNX模型简化
- 跨平台推理测试

作者: 破夜绘明团队
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time

import numpy as np

# PyTorch导入
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ONNX导入
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ONNX Runtime导入
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 导出配置
# =============================================================================
@dataclass
class ExportConfig:
    """ONNX导出配置"""
    opset_version: int = 17
    dynamic_batch: bool = True
    dynamic_height: bool = False
    dynamic_width: bool = False
    simplify: bool = True
    fp16: bool = False
    verify: bool = True
    verbose: bool = False


# =============================================================================
# ONNX导出器
# =============================================================================
class ONNXExporter:
    """
    ONNX模型导出器
    
    支持:
    - 标准PyTorch模型导出
    - 动态维度支持
    - 模型简化
    - FP16量化
    - 导出验证
    """
    
    def __init__(self, opset_version: int = 17, config: ExportConfig = None):
        """
        初始化导出器
        
        Args:
            opset_version: ONNX opset版本
            config: 导出配置
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装")
        
        self.config = config or ExportConfig(opset_version=opset_version)
    
    def export(self, model: nn.Module, input_shape: Tuple[int, ...],
               save_path: str, input_names: List[str] = None,
               output_names: List[str] = None,
               dynamic_batch: bool = None) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状 (C, H, W) 不含batch
            save_path: 保存路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_batch: 是否支持动态batch
        
        Returns:
            ONNX文件路径
        """
        logger.info(f"开始导出ONNX模型: {save_path}")
        
        # 准备模型
        model.eval()
        model = model.to("cpu")
        
        # 创建dummy输入
        dummy_input = torch.randn(1, *input_shape)
        
        # 设置动态轴
        dynamic_axes = {}
        if dynamic_batch or self.config.dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        if self.config.dynamic_height:
            dynamic_axes.setdefault('input', {})[2] = 'height'
        if self.config.dynamic_width:
            dynamic_axes.setdefault('input', {})[3] = 'width'
        
        # 输入输出名称
        input_names = input_names or ['input']
        output_names = output_names or ['output']
        
        # 创建保存目录
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 导出
        try:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                opset_version=self.config.opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes if dynamic_axes else None,
                do_constant_folding=True,
                verbose=self.config.verbose
            )
            logger.info(f"✅ ONNX导出成功: {save_path}")
            
        except Exception as e:
            logger.error(f"❌ ONNX导出失败: {e}")
            raise
        
        # 验证
        if self.config.verify:
            self._verify_onnx(save_path, dummy_input)
        
        # 简化
        if self.config.simplify:
            self._simplify_onnx(save_path)
        
        # FP16转换
        if self.config.fp16:
            self._convert_fp16(save_path)
        
        # 记录模型信息
        self._save_model_info(save_path, input_shape)
        
        return save_path
    
    def _verify_onnx(self, onnx_path: str, dummy_input: torch.Tensor):
        """验证ONNX模型"""
        if not ONNX_AVAILABLE:
            logger.warning("onnx未安装,跳过验证")
            return
        
        logger.info("验证ONNX模型...")
        
        # 检查模型结构
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("✅ ONNX模型结构验证通过")
        
        # 推理测试
        if ORT_AVAILABLE:
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            numpy_input = dummy_input.numpy()
            output = session.run(None, {input_name: numpy_input})
            
            logger.info(f"✅ ONNX推理测试通过, 输出形状: {[o.shape for o in output]}")
    
    def _simplify_onnx(self, onnx_path: str):
        """简化ONNX模型"""
        try:
            import onnxsim
            
            logger.info("简化ONNX模型...")
            
            model = onnx.load(onnx_path)
            simplified, ok = onnxsim.simplify(model)
            
            if ok:
                onnx.save(simplified, onnx_path)
                
                # 比较大小
                original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
                logger.info(f"✅ ONNX模型简化完成, 大小: {original_size:.2f} MB")
            else:
                logger.warning("⚠️ ONNX简化失败,保留原始模型")
                
        except ImportError:
            logger.info("onnxsim未安装,跳过简化步骤")
        except Exception as e:
            logger.warning(f"ONNX简化出错: {e}")
    
    def _convert_fp16(self, onnx_path: str):
        """转换为FP16"""
        try:
            from onnxconverter_common import float16
            
            logger.info("转换为FP16...")
            
            model = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(model)
            
            fp16_path = onnx_path.replace('.onnx', '_fp16.onnx')
            onnx.save(model_fp16, fp16_path)
            
            logger.info(f"✅ FP16模型已保存: {fp16_path}")
            
        except ImportError:
            logger.info("onnxconverter-common未安装,跳过FP16转换")
        except Exception as e:
            logger.warning(f"FP16转换出错: {e}")
    
    def _save_model_info(self, onnx_path: str, input_shape: Tuple):
        """保存模型信息"""
        info = {
            "onnx_path": onnx_path,
            "opset_version": self.config.opset_version,
            "input_shape": list(input_shape),
            "dynamic_batch": self.config.dynamic_batch,
            "fp16": self.config.fp16,
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 获取文件大小
        if Path(onnx_path).exists():
            info["size_mb"] = Path(onnx_path).stat().st_size / (1024 * 1024)
        
        # 获取ONNX模型信息
        if ONNX_AVAILABLE:
            model = onnx.load(onnx_path)
            info["ir_version"] = model.ir_version
            info["producer"] = model.producer_name
            info["num_nodes"] = len(model.graph.node)
        
        # 保存信息
        info_path = onnx_path.replace('.onnx', '_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型信息已保存: {info_path}")


# =============================================================================
# ONNX验证器
# =============================================================================
class ONNXValidator:
    """
    ONNX模型验证器
    
    用于验证导出的ONNX模型:
    - 结构完整性检查
    - 推理一致性验证
    - 性能基准测试
    """
    
    def __init__(self, onnx_path: str, use_gpu: bool = False):
        """
        初始化验证器
        
        Args:
            onnx_path: ONNX模型路径
            use_gpu: 是否使用GPU
        """
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime未安装")
        
        self.onnx_path = onnx_path
        self.use_gpu = use_gpu
        
        # 创建推理会话
        self.session = self._create_session()
        
        # 获取输入输出信息
        self.input_info = self._get_input_info()
        self.output_info = self._get_output_info()
    
    def _create_session(self) -> ort.InferenceSession:
        """创建推理会话"""
        providers = []
        
        if self.use_gpu:
            # 尝试TensorRT
            if 'TensorrtExecutionProvider' in ort.get_available_providers():
                providers.append(('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_fp16_enable': True,
                    'trt_max_workspace_size': 2 * 1024 * 1024 * 1024
                }))
            
            # CUDA
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                }))
        
        # CPU后备
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info(f"推理提供者: {session.get_providers()}")
        
        return session
    
    def _get_input_info(self) -> List[Dict]:
        """获取输入信息"""
        inputs = []
        for inp in self.session.get_inputs():
            inputs.append({
                'name': inp.name,
                'shape': inp.shape,
                'type': inp.type
            })
        return inputs
    
    def _get_output_info(self) -> List[Dict]:
        """获取输出信息"""
        outputs = []
        for out in self.session.get_outputs():
            outputs.append({
                'name': out.name,
                'shape': out.shape,
                'type': out.type
            })
        return outputs
    
    def validate_structure(self) -> Dict[str, Any]:
        """验证模型结构"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not ONNX_AVAILABLE:
            results["warnings"].append("onnx库未安装,跳过结构验证")
            return results
        
        try:
            model = onnx.load(self.onnx_path)
            onnx.checker.check_model(model)
            
            results["ir_version"] = model.ir_version
            results["opset_version"] = model.opset_import[0].version
            results["num_nodes"] = len(model.graph.node)
            results["num_inputs"] = len(model.graph.input)
            results["num_outputs"] = len(model.graph.output)
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(str(e))
        
        return results
    
    def validate_inference(self, pytorch_model: nn.Module = None,
                          input_data: np.ndarray = None,
                          rtol: float = 1e-3, atol: float = 1e-5) -> Dict[str, Any]:
        """
        验证推理一致性
        
        Args:
            pytorch_model: PyTorch模型(用于对比)
            input_data: 输入数据
            rtol: 相对容差
            atol: 绝对容差
        
        Returns:
            验证结果
        """
        results = {
            "inference_ok": True,
            "consistency_ok": None,
            "errors": []
        }
        
        # 生成或使用提供的输入
        if input_data is None:
            input_shape = self.input_info[0]['shape']
            # 处理动态维度
            input_shape = [s if isinstance(s, int) else 1 for s in input_shape]
            input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # ONNX推理
        try:
            input_name = self.input_info[0]['name']
            onnx_output = self.session.run(None, {input_name: input_data})
            results["onnx_output_shapes"] = [o.shape for o in onnx_output]
        except Exception as e:
            results["inference_ok"] = False
            results["errors"].append(f"ONNX推理失败: {e}")
            return results
        
        # PyTorch对比
        if pytorch_model is not None and TORCH_AVAILABLE:
            try:
                pytorch_model.eval()
                with torch.no_grad():
                    pytorch_input = torch.from_numpy(input_data)
                    pytorch_output = pytorch_model(pytorch_input)
                    
                    if isinstance(pytorch_output, torch.Tensor):
                        pytorch_output = pytorch_output.numpy()
                    elif isinstance(pytorch_output, dict):
                        pytorch_output = list(pytorch_output.values())[0]
                        if isinstance(pytorch_output, torch.Tensor):
                            pytorch_output = pytorch_output.numpy()
                
                # 比较输出
                is_close = np.allclose(onnx_output[0], pytorch_output, rtol=rtol, atol=atol)
                results["consistency_ok"] = is_close
                
                if not is_close:
                    max_diff = np.max(np.abs(onnx_output[0] - pytorch_output))
                    results["max_difference"] = float(max_diff)
                    results["errors"].append(f"输出不一致,最大差异: {max_diff}")
                
            except Exception as e:
                results["errors"].append(f"PyTorch对比失败: {e}")
        
        return results
    
    def benchmark(self, input_shape: Tuple = None, 
                  num_iterations: int = 100,
                  warmup: int = 10) -> Dict[str, Any]:
        """
        性能基准测试
        
        Args:
            input_shape: 输入形状
            num_iterations: 测试迭代次数
            warmup: 预热次数
        
        Returns:
            性能统计
        """
        # 准备输入
        if input_shape is None:
            input_shape = self.input_info[0]['shape']
            input_shape = [s if isinstance(s, int) else 1 for s in input_shape]
        
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input_name = self.input_info[0]['name']
        
        # 预热
        for _ in range(warmup):
            self.session.run(None, {input_name: input_data})
        
        # 测试
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.session.run(None, {input_name: input_data})
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # 转为毫秒
        
        return {
            "input_shape": list(input_shape),
            "num_iterations": num_iterations,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "fps": float(1000 / np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }
    
    def full_validation(self, pytorch_model: nn.Module = None) -> Dict[str, Any]:
        """完整验证"""
        results = {
            "onnx_path": self.onnx_path,
            "providers": self.session.get_providers(),
            "input_info": self.input_info,
            "output_info": self.output_info,
        }
        
        # 结构验证
        results["structure"] = self.validate_structure()
        
        # 推理验证
        results["inference"] = self.validate_inference(pytorch_model)
        
        # 性能测试
        results["benchmark"] = self.benchmark()
        
        # 总体状态
        results["status"] = "PASS" if (
            results["structure"]["valid"] and 
            results["inference"]["inference_ok"]
        ) else "FAIL"
        
        return results


# =============================================================================
# 便捷函数
# =============================================================================
def export_to_onnx(model: nn.Module, input_shape: Tuple,
                   save_path: str, **kwargs) -> str:
    """
    导出模型为ONNX
    
    Example:
        export_to_onnx(model, (3, 640, 640), "model.onnx")
    """
    exporter = ONNXExporter()
    return exporter.export(model, input_shape, save_path, **kwargs)


def verify_onnx_model(onnx_path: str, pytorch_model: nn.Module = None,
                      use_gpu: bool = False) -> Dict:
    """
    验证ONNX模型
    
    Example:
        results = verify_onnx_model("model.onnx", pytorch_model)
        print(f"状态: {results['status']}")
    """
    validator = ONNXValidator(onnx_path, use_gpu=use_gpu)
    return validator.full_validation(pytorch_model)


def benchmark_onnx(onnx_path: str, input_shape: Tuple = None,
                   use_gpu: bool = False) -> Dict:
    """
    ONNX模型性能测试
    
    Example:
        stats = benchmark_onnx("model.onnx", use_gpu=True)
        print(f"FPS: {stats['fps']:.1f}")
    """
    validator = ONNXValidator(onnx_path, use_gpu=use_gpu)
    return validator.benchmark(input_shape)


# =============================================================================
# 批量导出工具
# =============================================================================
class BatchExporter:
    """
    批量ONNX导出工具
    
    用于一次性导出所有插件模型
    """
    
    def __init__(self, checkpoint_dir: str, output_dir: str):
        """
        初始化批量导出器
        
        Args:
            checkpoint_dir: 检查点目录
            output_dir: ONNX输出目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.exporter = ONNXExporter()
    
    def export_all(self, model_configs: Dict[str, Dict]) -> Dict[str, str]:
        """
        批量导出所有模型
        
        Args:
            model_configs: 模型配置字典
                {
                    "transformer/defect_yolov8n": {
                        "model_type": "detection",
                        "input_size": (640, 640),
                        "checkpoint": "transformer/defect_yolov8n_best.pth"
                    },
                    ...
                }
        
        Returns:
            导出结果字典 {模型名: ONNX路径}
        """
        from .models import create_model
        
        results = {}
        
        for model_key, config in model_configs.items():
            logger.info(f"\n导出模型: {model_key}")
            
            try:
                # 解析配置
                plugin_name, model_name = model_key.split('/')
                model_type = config.get('model_type', 'detection')
                input_size = config.get('input_size', (640, 640))
                checkpoint_path = self.checkpoint_dir / config.get('checkpoint', f"{model_key}_best.pth")
                
                # 创建模型
                model = create_model(
                    model_type=model_type,
                    model_name=model_name,
                    input_size=input_size,
                    pretrained=False,
                    plugin_name=plugin_name
                )
                
                # 加载权重
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    logger.info(f"已加载检查点: {checkpoint_path}")
                else:
                    logger.warning(f"检查点不存在,使用随机权重: {checkpoint_path}")
                
                # 导出
                output_path = self.output_dir / plugin_name / f"{model_name}.onnx"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.exporter.export(
                    model=model,
                    input_shape=(3, *input_size),
                    save_path=str(output_path)
                )
                
                results[model_key] = str(output_path)
                logger.info(f"✅ {model_key} -> {output_path}")
                
            except Exception as e:
                logger.error(f"❌ {model_key} 导出失败: {e}")
                results[model_key] = f"ERROR: {e}"
        
        # 保存导出摘要
        summary_path = self.output_dir / "export_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results


# =============================================================================
# Windows验证脚本生成器
# =============================================================================
def generate_windows_validation_script(onnx_dir: str, output_path: str):
    """
    生成Windows环境验证脚本
    
    Args:
        onnx_dir: ONNX模型目录
        output_path: 脚本输出路径
    """
    script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows环境ONNX模型验证脚本
自动生成于Mac开发环境

使用方法:
    python validate_onnx_windows.py
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("请安装onnxruntime: pip install onnxruntime-gpu")
    sys.exit(1)

# 模型目录
ONNX_DIR = "{onnx_dir}"

def validate_model(onnx_path, use_gpu=True):
    """验证单个模型"""
    print(f"\\n验证模型: {{onnx_path}}")
    
    # 选择提供者
    providers = []
    if use_gpu:
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    # 创建会话
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"  使用提供者: {{session.get_providers()}}")
    except Exception as e:
        print(f"  ❌ 加载失败: {{e}}")
        return False, None
    
    # 获取输入信息
    input_info = session.get_inputs()[0]
    input_shape = [s if isinstance(s, int) else 1 for s in input_info.shape]
    
    # 生成测试输入
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 推理测试
    try:
        start = time.perf_counter()
        output = session.run(None, {{input_info.name: test_input}})
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"  ✅ 推理成功")
        print(f"  输入形状: {{input_shape}}")
        print(f"  输出形状: {{[o.shape for o in output]}}")
        print(f"  推理时间: {{inference_time:.2f}} ms")
        
        return True, inference_time
    except Exception as e:
        print(f"  ❌ 推理失败: {{e}}")
        return False, None

def benchmark_model(onnx_path, num_iterations=100, use_gpu=True):
    """性能测试"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_info = session.get_inputs()[0]
    input_shape = [s if isinstance(s, int) else 1 for s in input_info.shape]
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 预热
    for _ in range(10):
        session.run(None, {{input_info.name: test_input}})
    
    # 测试
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {{input_info.name: test_input}})
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    return {{
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "fps": float(1000 / np.mean(times))
    }}

def main():
    print("=" * 60)
    print("Windows ONNX模型验证")
    print("=" * 60)
    
    # 检测GPU
    providers = ort.get_available_providers()
    use_gpu = 'CUDAExecutionProvider' in providers
    print(f"\\n可用提供者: {{providers}}")
    print(f"使用GPU: {{use_gpu}}")
    
    # 查找所有ONNX模型
    onnx_dir = Path(ONNX_DIR)
    onnx_files = list(onnx_dir.rglob("*.onnx"))
    
    print(f"\\n找到 {{len(onnx_files)}} 个ONNX模型")
    
    results = {{}}
    for onnx_path in onnx_files:
        success, time_ms = validate_model(str(onnx_path), use_gpu)
        
        model_name = str(onnx_path.relative_to(onnx_dir))
        results[model_name] = {{
            "status": "PASS" if success else "FAIL",
            "inference_time_ms": time_ms
        }}
        
        if success:
            # 性能测试
            stats = benchmark_model(str(onnx_path), use_gpu=use_gpu)
            results[model_name].update(stats)
    
    # 保存结果
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    print("\\n" + "=" * 60)
    print("验证摘要")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = len(results) - passed
    
    print(f"通过: {{passed}}")
    print(f"失败: {{failed}}")
    
    if failed > 0:
        print("\\n失败的模型:")
        for name, result in results.items():
            if result["status"] == "FAIL":
                print(f"  - {{name}}")

if __name__ == "__main__":
    main()
'''.format(onnx_dir=onnx_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    logger.info(f"Windows验证脚本已生成: {output_path}")
