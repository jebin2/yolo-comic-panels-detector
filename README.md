
# YOLO Comic Panels Detector

A lightweight deep learning project that uses YOLOv11 to train and detect panels from comic pages.

## 🚀 Features
- Trainable on custom comic datasets
- YOLOv11-based panel detection

## 🖼️ Example Output
Annotated comic panels with bounding boxes:

![Example](annotated.png)

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
