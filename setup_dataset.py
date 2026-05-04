import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import zipfile
import shutil
import json
from config.settings import DATASET_PATH, CLASSES


PLANTVILLAGE_KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"
PLANTVILLAGE_KAGGLE_CLASSES = {
    "Potato___healthy": "healthy_potato",
    "Potato___Late_blight": "potato_blight",
}


def download_and_setup_kaggle(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    print("=" * 60)
    print("Downloading Real Plant Disease Dataset from Kaggle")
    print("=" * 60)

    try:
        import kagglehub
    except ImportError:
        print("\nkagglehub not installed. Installing now...")
        os.system("pip install kagglehub")
        import kagglehub

    print("\nDownloading New Plant Diseases Dataset (this may take a few minutes)...")
    print("Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")

    try:
        path = kagglehub.dataset_download(PLANTVILLAGE_KAGGLE_DATASET)
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"\nKaggle download failed: {e}")
        print("\nFalling back to direct download method...")
        return download_and_setup_direct(dataset_path)

    source_base = os.path.join(path, "New Plant Diseases Dataset(Augmented)")
    if not os.path.exists(source_base):
        source_base = path

    train_dir = os.path.join(source_base, "train")
    if not os.path.exists(train_dir):
        print(f"\nExpected train directory not found at: {train_dir}")
        print(f"Available directories: {os.listdir(source_base) if os.path.exists(source_base) else 'N/A'}")
        return download_and_setup_direct(dataset_path)

    print(f"\nFound dataset at: {train_dir}")
    print("Organizing images into project structure...")

    for class_name in CLASSES:
        class_dir = os.path.join(dataset_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

    files_copied = 0
    for kaggle_class, our_class in PLANTVILLAGE_KAGGLE_CLASSES.items():
        kaggle_dir = os.path.join(train_dir, kaggle_class)
        if not os.path.exists(kaggle_dir):
            print(f"  Warning: Class '{kaggle_class}' not found in downloaded dataset")
            continue

        target_dir = os.path.join(dataset_path, our_class)
        for filename in os.listdir(kaggle_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(kaggle_dir, filename)
                dst = os.path.join(target_dir, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    files_copied += 1

        count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {our_class}: {count} images")

    print(f"\nTotal files copied: {files_copied}")
    print(f"Dataset ready at: {dataset_path}")
    return dataset_path


def download_and_setup_direct(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    print("=" * 60)
    print("Downloading PlantVillage Dataset (Direct)")
    print("=" * 60)
    print("\nThis will download ~200MB of real leaf images.")
    print("If this fails, please download manually from:")
    print("  https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("  or https://github.com/spMohanty/PlantVillage-Dataset")
    print()

    urls = {
        "healthy_potato": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/healthy/Potato___healthy/",
        "healthy_tomato": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/healthy/Tomato___healthy/",
        "potato_blight": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Late_blight/Potato___Late_blight/",
        "tomato_blight": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Late_blight/Tomato___Late_blight/",
    }

    all_downloaded = True
    for class_name, base_url in urls.items():
        class_dir = os.path.join(dataset_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        existing = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if existing:
            print(f"  {class_name}: {len(existing)} images already present, skipping download")
            continue

        print(f"  Downloading {class_name} images...")
        try:
            list_url = base_url.rstrip("/") + "?ref=badge_small"
            req = urllib.request.Request(list_url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=15)
            html = response.read().decode("utf-8", errors="ignore")

            import re
            image_links = re.findall(r'href="([^"]+\.JPG)"', html)
            image_links = list(set(image_links))[:500]

            if not image_links:
                image_links = re.findall(r'href="([^"]+\.(?:jpg|jpeg|png))"', html, re.IGNORECASE)
                image_links = list(set(image_links))[:500]

            downloaded = 0
            for link in image_links:
                if link.startswith("http"):
                    img_url = link
                else:
                    img_url = base_url.rstrip("/") + "/" + link.split("/")[-1]

                filename = link.split("/")[-1]
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                dst = os.path.join(class_dir, filename)
                if os.path.exists(dst):
                    downloaded += 1
                    continue

                try:
                    req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                    urllib.request.urlretrieve(req.full_url if hasattr(req, "full_url") else img_url, dst)
                    downloaded += 1
                except Exception:
                    pass

            print(f"    {class_name}: {downloaded} images downloaded")
        except Exception as e:
            print(f"    {class_name}: Download failed ({e})")
            all_downloaded = False

    if not all_downloaded:
        print("\nSome downloads failed. Setting up with kagglehub instructions...")
        return setup_with_instructions(dataset_path)

    total = sum(len([f for f in os.listdir(os.path.join(dataset_path, c)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) for c in CLASSES)
    print(f"\nDataset ready at: {dataset_path} ({total} total images)")
    return dataset_path


def setup_with_instructions(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    print("\n" + "=" * 60)
    print("Manual Dataset Setup Instructions")
    print("=" * 60)
    print()
    print("Option 1: Using kagglehub (Recommended)")
    print("  1. Run: pip install kagglehub")
    print("  2. The script will auto-download on next run")
    print()
    print("Option 2: Download from Kaggle")
    print("  1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("  2. Download and extract the dataset")
    print("  3. Copy images from these folders to the project:")
    print()

    kaggle_to_our = {
        "New Plant Diseases Dataset(Augmented)/train/Potato___healthy": "dataset/plant_dataset/healthy_potato/",
        "New Plant Diseases Dataset(Augmented)/train/Tomato___healthy": "dataset/plant_dataset/healthy_tomato/",
        "New Plant Diseases Dataset(Augmented)/train/Potato___Late_blight": "dataset/plant_dataset/potato_blight/",
        "New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight": "dataset/plant_dataset/tomato_blight/",
    }

    for src, dst in kaggle_to_our.items():
        print(f"     {src}/  -->  {dst}")

    print()
    print("Option 3: Use PlantVillage from GitHub")
    print("  1. Go to: https://github.com/spMohanty/PlantVillage-Dataset")
    print("  2. Clone or download raw/color/ folder")
    print("  3. Map: raw/color/healthy/Potato___healthy/ --> dataset/plant_dataset/healthy_potato/")
    print("     raw/color/healthy/Tomato___healthy/ --> dataset/plant_dataset/healthy_tomato/")
    print("     raw/color/Late_blight/Potato___Late_blight/ --> dataset/plant_dataset/potato_blight/")
    print("     raw/color/Late_blight/Tomato___Late_blight/ --> dataset/plant_dataset/tomato_blight/")
    print()
    print("=" * 60)

    for class_name in CLASSES:
        class_dir = os.path.join(dataset_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {class_name}: {count} images found")

    return dataset_path


def verify_dataset(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_PATH

    print("\nDataset Verification")
    print("-" * 40)

    total = 0
    for class_name in CLASSES:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"  MISSING: {class_name}/")
            continue

        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        total += count
        status = "OK" if count >= 100 else "LOW (need 100+)"
        print(f"  {class_name}: {count} images [{status}]")

    print(f"\nTotal: {total} images")

    if total >= 400:
        print("Dataset is ready for training!")
        return True
    else:
        print("WARNING: Not enough images. Need at least 100 per class (400 total).")
        print("Use 'python setup_dataset.py --download' to get real images.")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup Leaf Blight Dataset")
    parser.add_argument("--download", action="store_true", help="Download real plant disease images")
    parser.add_argument("--kaggle", action="store_true", help="Download from Kaggle using kagglehub")
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset")
    parser.add_argument("--instructions", action="store_true", help="Show manual download instructions")

    args = parser.parse_args()

    if args.download or args.kaggle:
        download_and_setup_kaggle()
        verify_dataset()
    elif args.verify:
        verify_dataset()
    elif args.instructions:
        setup_with_instructions()
    else:
        print("Leaf Blight Dataset Setup")
        print("=" * 40)
        print()
        print("Commands:")
        print("  python setup_dataset.py --download      Download real dataset (recommended)")
        print("  python setup_dataset.py --kaggle        Download from Kaggle")
        print("  python setup_dataset.py --verify        Check current dataset")
        print("  python setup_dataset.py --instructions  Show manual setup guide")
        print()
        verify_dataset()
