# NEU-DET 数据集说明

## 数据集简介

NEU-DET（Northeastern University Surface Defect Database）是一个热轧钢带表面缺陷检测公开数据集，包含 6 类缺陷：

| 英文名称 | 简称 | 中文 |
|---|---|---|
| crazing | Cr | 裂纹 |
| inclusion | In | 夹杂 |
| patches | Pa | 斑块 |
| pitted_surface | Ps | 麻点 |
| rolled-in_scale | Rs | 氧化铁皮压入 |
| scratches | Sc | 划痕 |

## 标注格式

原始 NEU-DET 使用 XML 格式标注（PASCAL VOC 风格），包含每个缺陷的 bounding box 坐标 `(xmin, ymin, xmax, ymax)` 和类别标签。

本项目通过 `scripts/convert_annotations.py` 将其转换为 **YOLO 格式**：

```text
<class_id> <x_center> <y_center> <width> <height>
```

所有数值均归一化到 `[0, 1]`。

## 数据划分

默认按 8:1:1 划分为：

- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

每份子目录下包含 `images/` 和 `labels/`。

## Demo 数据说明

如果 NEU-DET 公开下载链接不可用，可使用 `scripts/generate_synthetic_data.py` 生成合成数据作为 fallback，以快速跑通训练、评估、ONNX 导出和推理 demo。合成数据包含 6 类缺陷的模拟样本，结构与真实 NEU-DET 一致。
