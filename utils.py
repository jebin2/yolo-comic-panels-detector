# utils.py
import os
import shutil
from glob import glob
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class to manage environment variables and paths."""
    YOLO_MODEL_NAME = os.getenv('YOLO_MODEL_NAME', 'default_model')
    DEFAULT_IMAGE_SIZE = 640
    SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

def get_abs_path(relative_path: str) -> str:
    """Convert relative path to absolute path."""
    return os.path.abspath(relative_path)

def get_image_paths(directories: Union[str, List[str]]) -> List[str]:
    """
    Get all image paths from given directories.
    
    Args:
        directories: Single directory path or list of directory paths
        
    Returns:
        List of image file paths
    """
    if isinstance(directories, str):
        directories = [directories]
    
    all_images = []
    for directory in directories:
        abs_dir = get_abs_path(directory)
        if not os.path.isdir(abs_dir):
            print(f"⚠️ Warning: Skipping non-directory {abs_dir}")
            continue
            
        # Support multiple image extensions
        for ext in Config.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(abs_dir, f'*.{ext}')
            images = sorted(glob(pattern))
            all_images.extend(images)
    
    return list(set(all_images))  # Remove duplicates

def clean_directory(directory: str, create_if_not_exists: bool = True) -> None:
    """Clean directory contents or create if it doesn't exist."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    if create_if_not_exists:
        os.makedirs(directory, exist_ok=True)

def backup_file(source_path: str, backup_path: str) -> str:
    """Backup a file to specified location."""
    backup_path = get_abs_path(backup_path)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy(source_path, backup_path)
    print(f"✅ File backed up to: {backup_path}")
    return backup_path