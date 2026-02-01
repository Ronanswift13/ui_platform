#!/bin/bash
# 室内监测中心功能测试脚本

echo "========================================="
echo "室内监测中心功能测试"
echo "========================================="
echo ""

BASE_URL="http://127.0.0.1:8080"

# 测试页面加载
echo "1. 测试页面加载..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/indoor)
if [ "$STATUS" = "200" ]; then
    echo "   ✓ 页面加载成功 (200 OK)"
else
    echo "   ✗ 页面加载失败 ($STATUS)"
fi
echo ""

# 测试融合证据链API
echo "2. 测试融合证据链API..."
FUSION=$(curl -s $BASE_URL/api/indoor/fusion/evidence)
CONFIDENCE=$(echo $FUSION | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['confidence']*100:.0f}%\")" 2>/dev/null)
RESULT=$(echo $FUSION | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['fusion_result'])" 2>/dev/null)
if [ -n "$CONFIDENCE" ]; then
    echo "   ✓ 融合证据链API正常"
    echo "     - 综合判定: $RESULT"
    echo "     - 置信度: $CONFIDENCE"
else
    echo "   ✗ 融合证据链API失败"
fi
echo ""

# 测试动态控制面板API
echo "3. 测试动态控制面板API..."
PLUGINS=("indoor_fence" "animal_detection" "temperature_monitoring")
PLUGIN_NAMES=("电子围栏" "动物入侵检测" "温度监测")

for i in "${!PLUGINS[@]}"; do
    PLUGIN="${PLUGINS[$i]}"
    NAME="${PLUGIN_NAMES[$i]}"
    CAP=$(curl -s $BASE_URL/api/indoor/plugin/$PLUGIN/capabilities)
    CONTROLS=$(echo $CAP | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('controls', [])))" 2>/dev/null)
    OPS=$(echo $CAP | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('operations', [])))" 2>/dev/null)

    if [ -n "$CONTROLS" ]; then
        echo "   ✓ $NAME"
        echo "     - 控制参数: $CONTROLS 个"
        echo "     - 操作按钮: $OPS 个"
    else
        echo "   ✗ $NAME API失败"
    fi
done
echo ""

# 测试监测模块API
echo "4. 测试监测模块API..."
MODULES=("fence" "animal" "temperature" "device" "fire" "environment")
MODULE_NAMES=("电子围栏" "动物入侵" "温度监测" "设备状态" "消防监测" "环境监测")

for i in "${!MODULES[@]}"; do
    MODULE="${MODULES[$i]}"
    NAME="${MODULE_NAMES[$i]}"
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/api/indoor/$MODULE)

    if [ "$STATUS_CODE" = "200" ]; then
        echo "   ✓ $NAME API (200 OK)"
    else
        echo "   ✗ $NAME API ($STATUS_CODE)"
    fi
done
echo ""

# 测试服务器状态
echo "5. 服务器状态..."
PID=$(ps aux | grep "ui_server.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "   ✓ 服务器运行中 (PID: $PID)"
    echo "   ✓ 端口: 8080"
    echo "   ✓ 地址: http://127.0.0.1:8080/indoor"
else
    echo "   ✗ 服务器未运行"
fi
echo ""

echo "========================================="
echo "测试完成"
echo "========================================="
echo ""
echo "访问地址: http://127.0.0.1:8080/indoor"
echo ""
