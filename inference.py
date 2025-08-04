import os
import cv2
from glob import glob
from ultralytics import YOLO

def get_abs_path(relative_path):
    return os.path.abspath(relative_path)

def get_image_paths(directories):
    all_images = []
    for directory in directories:
        abs_dir = get_abs_path(directory)
        if not os.path.isdir(abs_dir):
            print(f"⚠️ Warning: Skipping non-directory {abs_dir}")
            continue
        images = sorted(glob(os.path.join(abs_dir, '*.[jp][pn]g')))
        all_images.extend(images)
    return all_images

def annotate_images(model, image_paths, output_dir='temp_dir', image_size=640):
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    else:
        os.makedirs(output_dir)

    for idx, image_path in enumerate(image_paths):
        print(f'🔍 Processing: {image_path}')
        results = model(image_path, imgsz=image_size)
        annotated_frame = results[0].plot()
        save_path = os.path.join(output_dir, f'annotated_{idx}.png')
        cv2.imwrite(save_path, annotated_frame)
        print(f'✅ Saved: {save_path}')

def run_inference(weights_path, images_dirs, output_dir='temp_dir'):
    weights_path = get_abs_path(weights_path)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"❌ Weights file not found: {weights_path}")

    # Normalize input to list
    if isinstance(images_dirs, str):
        images_dirs = [images_dirs]

    image_paths = get_image_paths(images_dirs)
    if not image_paths:
        raise ValueError("❌ No images found in the provided directories.")

    model = YOLO(weights_path)
    annotate_images(model, image_paths, output_dir=output_dir)

if __name__ == "__main__":
    run_inference(
        weights_path='./comic_yolov8n_best.pt',
        images_dirs=['./dataset/images/train', './dataset/images/val', './dataset/images/test'],
        output_dir='./temp_dir'
    )
