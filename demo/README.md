# Demo Dataset (小型演示数据集)

此目录包含一个小的数据集子集（20 张图像），可用于测试代码。  
每张图像都包含完整的标注 — 检测边界框和分割多边形/掩码。

This directory contains a small subset (20 images) for testing the code.
Each image has complete annotations — both detection bboxes and segmentation polygons/masks.

## 结构 / Structure

```
demo/
├── detection/
│   ├── train/          # 5 张训练图像，附带 YOLO bbox 标注
│   │   ├── 0001.png, 0001.txt
│   │   └── ...
│   └── test/           # 20 张测试图像，附带 YOLO bbox 标注
│
├── segmentation/
│   ├── train/          # 5 张训练图像的分割标注
│   └── test/           # 20 张测试图像，附带 YOLO 多边形标注 + PNG 掩码
│
└── samples/            # 示例可视化输出
```

## 标注格式 / Annotation Formats

- **detection/**: YOLO bbox format — `class_id x_center y_center width height` (归一化 [0,1])
- **segmentation/**: YOLO polygon format — `class_id x1 y1 x2 y2 ...` (归一化 [0,1])

## 完整数据集 / Full Dataset

完整的数据集包含 5000 张裂缝图像和 5000 张无裂缝图像，可从以下位置获取：
The full dataset with 5000 crack images and 5000 non-crack images is available at:

**[数据集仓库链接 / Dataset Repository — 待添加]**
