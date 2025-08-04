
# YOLO Comic Panels Detector

A lightweight deep learning project that uses YOLOv8 to detect comic or manga panels from scanned pages.

## 🚀 Features
- Trainable on custom comic datasets
- YOLOv8-based panel detection
- Exportable to ONNX or TorchScript
- Batch inference with automatic annotation
- Easy integration for downstream tasks (e.g., OCR, captioning)

## 🖼️ Example Output
Annotated comic panels with bounding boxes:

![Example](temp_dir/annotated_0.png)

## 📁 Dataset Format
Defined in a `comic.yaml` file:

```yaml
train: dataset/images/train
val: dataset/images/val
nc: 1
names: ['panel']
````

## 🏁 Quick Start

```bash
# Clone the repo
git clone https://github.com/your-username/yolo-comic-panels-detector
cd yolo-comic-panels-detector

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
```

## 📂 Output Structure

After training:

```
runs/detect/comic_panel_yolov8n/
└── weights/
    ├── best.pt
    └── last.pt
```
