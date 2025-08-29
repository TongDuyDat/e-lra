import os
from pathlib import Path
import shutil


def load_file(path: str):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def cp_file(file_path, target_path):
    target_path_images = os.path.join(target_path, "images")
    target_path_masks = os.path.join(target_path, "masks")
    os.makedirs(target_path_images, exist_ok=True)
    os.makedirs(target_path_masks, exist_ok=True)
    for file in file_path:
        split_file = file.split("\\")
        if split_file[-2] == "images":
            dest = Path(
                os.path.join(target_path_images, f"{split_file[-1]}")
            ).with_suffix(".jpg")
            shutil.copyfile(file, dest)
        if split_file[-2] == "masks":
            dest = Path(
                os.path.join(target_path_masks, f"{split_file[-1].replace('p', '')}")
            ).with_suffix(".jpg")
            shutil.copyfile(file, dest)

        # shutil.copy(file, target_path)

