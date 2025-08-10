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

## 下载方式

NEU-DET 有多个公开下载源，按推荐顺序如下：

### 1. 官方主页（Google Drive / 百度网盘）

- 主页：http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
- 选择 **NEU-DET** 下载，推荐 Google Drive；国内网络可使用百度网盘，提取码 `pmqx`。

### 2. Kaggle 镜像

如果官方 Google Drive 出现配额/权限问题，可使用 Kaggle 镜像：

```bash
# 安装 Kaggle CLI（若未安装）
pip install kaggle

# 配置 API Token
# 1. 登录 https://www.kaggle.com/ -> Account -> Create New Token
# 2. 将下载的 kaggle.json 放到 ~/.kaggle/kaggle.json（Linux/macOS）
#    或 C:\Users\<你的用户名>\.kaggle\kaggle.json（Windows）
# 3. 确保文件权限安全：chmod 600 ~/.kaggle/kaggle.json

# 下载 NEU-DET
python scripts/download_neu_det.py --source kaggle
```

默认使用的 Kaggle 数据集是 `danielfinez/neu-det-steel-surface-defect-detection-dataset`，也可通过 `--kaggle-dataset` 指定其他镜像。

### 3. 手动放置

如果上述方式都不可用，可手动下载任意 NEU-DET 压缩包放到 `data/raw/`，然后执行：

```bash
python scripts/download_neu_det.py --source google-drive --force
# 或仅做解压
python scripts/convert_annotations.py
```

`convert_annotations.py` 会自动在 `data/raw/` 下递归查找 `images/` 和 `annotations/`（或 `Annotations/`）目录。

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
