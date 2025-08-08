import os
import shutil
from glob import glob
import cv2
from ultralytics import YOLO

def get_abs_path(relative_path):
    return os.path.abspath(relative_path)

def get_image_paths(directory):
    return sorted(glob(os.path.join(directory, '*.[jp][pn]g')))

def train_model(data_yaml_path, model_name='yolo11s.pt', run_name='comic_panel_yolov8n', device=0):
    model = YOLO(model_name)
    model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=200,
        batch=10,
        name=run_name,
        device=device,
        cache=True,
        project='runs/detect',
        exist_ok=True
    )
    return model

def validate_model(model):
    metrics = model.val()
    print("📊 Validation Metrics:", metrics)
    return metrics

def get_trained_weights(run_name):
    path = os.path.join('runs', 'detect', run_name, 'weights', 'best.pt')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ Trained weights not found at: {path}")
    return path

def backup_weights(source_path, backup_path='./comic_yolov8n_best.pt'):
    backup_path = os.path.abspath(backup_path)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy(source_path, backup_path)
    print(f"✅ Trained model backed up to: {backup_path}")
    return backup_path

def annotate_images(model, image_paths, output_dir='temp_dir', image_size=640):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for idx, image_path in enumerate(image_paths):
        print(f'🔍 Processing: {image_path}')
        results = model(image_path, imgsz=image_size)
        annotated_frame = results[0].plot()
        save_path = os.path.join(output_dir, f'annotated_{idx}.png')
        cv2.imwrite(save_path, annotated_frame)
        print(f'✅ Saved: {save_path}')


def main():
    # Config Paths
    data_yaml_path = get_abs_path('./comic.yaml')
    val_dir = get_abs_path('dataset/images/val')
    image_paths = get_image_paths(val_dir)

    # Train
    run_name = 'comic_panel_yolov8n'
    model = train_model(data_yaml_path, run_name=run_name)

    # Validate
    validate_model(model)

    # Save best.pt
    weights_path = get_trained_weights(run_name)
    backup_path = backup_weights(weights_path)

    # Reload model for inference
    model = YOLO(backup_path)

    # Annotate
    annotate_images(model, image_paths)


if __name__ == "__main__":
    main()
