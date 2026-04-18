# Terminal Execute Prompt — 表计读数

## 用途
在终端环境中执行表计读数相关的调试和测试命令。

## 常用命令模板

### 单张图片测试
```bash
cd plugins/meter_reading
python -c "
from detector_enhanced import MeterReadingDetectorEnhanced
import cv2
detector = MeterReadingDetectorEnhanced()
detector.initialize({})
img = cv2.imread('tests/fixtures/analog/pressure/sample_01.jpg')
result = detector.read_meter(img, 'pressure_gauge')
print(f'Value: {result.value}, Confidence: {result.confidence}, Status: {result.status}')
"
```

### 批量回归测试
```bash
scripts/run_regression_tests.sh
```

### 性能基准
```bash
python -c "
import time, cv2, numpy as np
from detector_enhanced import MeterReadingDetectorEnhanced
detector = MeterReadingDetectorEnhanced()
detector.initialize({})
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    detector.read_meter(img, 'pressure_gauge')
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(f'P50: {times[50]:.1f}ms, P95: {times[95]:.1f}ms, P99: {times[99]:.1f}ms')
"
```

### Standalone 启动
```bash
python run_standalone.py
# 浏览器打开 http://localhost:8091
```
