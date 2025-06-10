import os
from pathlib import Path

DATASET_DIR = "dataset/train"
SUBFOLDERS = ["inside", "outside"]

def rename_images_to_numbers():
    counter = 0
    for cls in SUBFOLDERS:
        folder_path = Path(DATASET_DIR) / cls
        for img_file in sorted(folder_path.iterdir()):
            if img_file.is_file():
                ext = img_file.suffix.lower()
                new_name = f"{counter:06d}{ext}"
                new_path = folder_path / new_name
                try:
                    img_file.rename(new_path)
                    counter += 1
                except Exception as e:
                    print(f"Failed to rename {img_file.name}: {e}")

    print(f"Renamed {counter} images successfully.")

if __name__ == "__main__":
    rename_images_to_numbers()
