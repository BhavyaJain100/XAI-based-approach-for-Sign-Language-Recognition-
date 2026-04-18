"""
Data Loader for isl_dataset with manual Train/Test split.

Dataset structure:
    isl_dataset/
        Train/
            A/  A_0.jpg ... A_700.jpg
            B/  ...
            ...
            9/  ...
        Test/
            A/  A_701.jpg ... A_999.jpg
            ...

Splits:
    Train folder  -> 80% train | 20% val  (stratified, random_state fixed)
    Test folder   -> 100% test  (never seen during training)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import config
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ISLDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Path collection  (works for both Train/ and Test/ folders)
# ─────────────────────────────────────────────────────────────────────────────

def collect_image_paths(dataset_dir):
    """
    Scan dataset_dir; each immediate sub-folder is a class.
    Returns (image_paths, labels).
    """
    image_paths, labels = [], []

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return [], []

    class_folders = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    if not class_folders:
        print(f"No sub-folders found in: {dataset_dir}")
        return [], []

    print(f"Found {len(class_folders)} class folders in {dataset_dir}")

    skipped = []
    for class_name in tqdm(class_folders, desc="Scanning classes"):
        class_path = os.path.join(dataset_dir, class_name)
        class_idx  = config.CLASS_TO_IDX.get(
            class_name,
            config.CLASS_TO_IDX.get(class_name.upper(), None)
        )
        if class_idx is None:
            skipped.append(class_name)
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(class_path, fname))
                labels.append(class_idx)

    if skipped:
        print(f"Skipped unknown class folders: {skipped}")

    return image_paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# SimpleDataLoader  — loads Train/ and Test/ separately
# ─────────────────────────────────────────────────────────────────────────────

class SimpleDataLoader:
    """
    Loads images from TRAIN_DIR and TEST_DIR separately.

    Usage (unchanged API):
        loader = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
        train_paths, train_labels, test_paths, test_labels = loader.load_data()

    Returns:
        train_paths  : all paths from Train/ folder (val split done later)
        train_labels : corresponding labels
        test_paths   : all paths from Test/ folder
        test_labels  : corresponding labels
    """

    def __init__(self, train_dir, test_dir=None):
        self.train_dir = train_dir
        self.test_dir  = test_dir or config.TEST_DIR

    def load_data(self):
        # ── Train folder ─────────────────────────────────────────────────────
        print(f"\nScanning Train folder: {self.train_dir}")
        train_paths, train_labels = collect_image_paths(self.train_dir)

        # ── Test folder ──────────────────────────────────────────────────────
        print(f"\nScanning Test folder: {self.test_dir}")
        test_paths, test_labels = collect_image_paths(self.test_dir)

        print(f"\nData Statistics:")
        print(f"  Train folder images : {len(train_paths)}")
        print(f"  Test  folder images : {len(test_paths)}")

        return train_paths, train_labels, test_paths, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(augment=True):
    """
    augment=True  -> training pipeline  (strong stochastic augmentation)
    augment=False -> val/test pipeline  (deterministic, no augmentation)
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT + 20, config.IMG_WIDTH + 20)),
            transforms.RandomCrop((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=0, translate=(0.12, 0.12),
                scale=(0.85, 1.15), shear=12),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0), value='random'),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])


# Backward-compat alias
get_simple_transforms = get_transforms


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def create_data_loaders(
    train_paths, train_labels,
    test_paths,  test_labels,
    validation_split=None,
):
    """
    Splits train_paths 80/20 into train/val, then builds three DataLoaders.

    Train folder -> 80% train | 20% val  (stratified)
    Test folder  -> 100% test
    """
    if validation_split is None:
        validation_split = config.VALIDATION_SPLIT   # 0.20

    # ── Train / Val split (from Train folder only) ───────────────────────────
    split_train_paths, split_val_paths, \
    split_train_labels, split_val_labels = train_test_split(
        train_paths, train_labels,
        test_size=validation_split,
        random_state=config.RANDOM_SEED,
        stratify=train_labels,
    )

    print(f"\n  Train  : {len(split_train_paths)} images")
    print(f"  Val    : {len(split_val_paths)} images")
    print(f"  Test   : {len(test_paths)} images")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = ISLDataset(split_train_paths, split_train_labels,
                               transform=get_transforms(augment=True))
    val_dataset   = ISLDataset(split_val_paths,   split_val_labels,
                               transform=get_transforms(augment=False))
    test_dataset  = ISLDataset(test_paths,         test_labels,
                               transform=get_transforms(augment=False))

    persistent = config.NUM_WORKERS > 0

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=persistent,
    )

    print(f"\nDataLoaders created:")
    print(f"  Training batches   : {len(train_loader)}")
    print(f"  Validation batches : {len(val_loader)}")
    print(f"  Test batches       : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# Backward-compat alias
create_simple_data_loaders = create_data_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
    train_paths, train_labels, test_paths, test_labels = loader.load_data()

    if train_paths:
        train_loader, val_loader, test_loader = create_data_loaders(
            train_paths, train_labels, test_paths, test_labels)
        for images, labels in train_loader:
            print(f"\nBatch test:")
            print(f"  Images shape : {images.shape}")
            print(f"  Labels shape : {labels.shape}")
            print(f"  Pixel range  : [{images.min():.3f}, {images.max():.3f}]")
            break
        print("\nData loading test successful!")
    else:
        print("No data loaded — check TRAIN_DIR and TEST_DIR in config.py")