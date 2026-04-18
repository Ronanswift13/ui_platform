#!/bin/bash
# 运行针对性测试
# 用法: ./run_targeted_tests.sh <module_name>

set -e

MODULE=${1:-"all"}

echo "=== 运行针对性测试: $MODULE ==="

case $MODULE in
  "fusion")
    echo "测试传感器融合模块..."
    pytest tests/test_fusion_v3.py tests/test_ekf.py -v --cov=core/fusion
    ;;

  "adapters")
    echo "测试硬件适配器..."
    pytest tests/test_uwb_adapter.py tests/test_imu_adapter.py tests/test_simulator.py -v --cov=adapters
    ;;

  "detection")
    echo "测试行为检测..."
    pytest tests/test_detection.py tests/test_model_manager.py -v --cov=detection
    ;;

  "rules")
    echo "测试规则引擎..."
    pytest tests/test_rules.py tests/test_auto_fence.py -v --cov=core/rules
    ;;

  "state_machine")
    echo "测试状态机..."
    pytest tests/test_state_machine_v3.py -v --cov=core/state_machine.py
    ;;

  "standalone")
    echo "测试独立运行模块..."
    pytest tests/test_data_recorder.py tests/test_data_replayer.py tests/test_training.py -v --cov=standalone
    ;;

  "integration")
    echo "测试集成场景..."
    pytest tests/test_integration.py -v --cov=indoor_fence
    ;;

  "all")
    echo "运行所有测试..."
    pytest tests/ -v --cov=indoor_fence --cov-report=html --cov-report=term
    ;;

  *)
    echo "错误: 未知模块 '$MODULE'"
    echo "可用模块: fusion, adapters, detection, rules, state_machine, standalone, integration, all"
    exit 1
    ;;
esac

echo ""
echo "=== 测试完成 ==="
echo "查看详细报告: open htmlcov/index.html"
