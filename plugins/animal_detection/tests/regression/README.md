# Regression Tests (L2)

## 状态
待标定数据集就绪后激活。

## 标定数据集结构

```
tests/regression/fixtures/
├── images/           # 标定图片 (jpg/png)
│   ├── mouse_001.jpg
│   ├── snake_001.jpg
│   └── ...
├── labels/           # YOLO 格式标注
│   ├── mouse_001.txt
│   ├── snake_001.txt
│   └── ...
└── baseline.json     # 精度基线记录
```

## baseline.json 格式

```json
{
  "version": "1.0.0",
  "date": "2026-04-08",
  "model": "animal_yolov8n.onnx",
  "metrics": {
    "overall": {"recall": 0.87, "precision": 0.83, "fpr": 0.03},
    "per_class": {
      "mouse": {"recall": 0.90, "precision": 0.85},
      "snake": {"recall": 0.82, "precision": 0.88}
    }
  }
}
```

## 激活步骤

1. 准备标定图片集（每类至少 20 张，含正例和负例）
2. 放入 `fixtures/images/` 和 `fixtures/labels/`
3. 运行首次评估生成 `baseline.json`
4. 删除 `test_precision_baseline.py` 中的 `pytest.skip` 行
5. 将 `BASELINE_EXISTS` 检查改为实际文件验证

## 运行

```bash
pytest tests/regression/ -v -m regression
```
