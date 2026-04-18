"""
Occlusion Sensitivity Explainability
Slides a grey patch across the image and measures how much the model's
confidence drops — the bigger the drop, the more important that region.

Works in two modes:
  1. Batch mode  : automatically picks samples from the test set (like gradcam/lime)
  2. Single image: pass an image path on the command line (like shap)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from datetime import datetime
from PIL import Image

import config
from data_loader import SimpleDataLoader, create_data_loaders, get_transforms
from model_architecture import create_resnet18_model


# ─────────────────────────────────────────────────────────────────────────────
# Core Occlusion Sensitivity class
# ─────────────────────────────────────────────────────────────────────────────

class OcclusionSensitivity:
    """
    Slides an occlusion patch over the image in a grid pattern.
    At each position we replace that patch with a solid grey square,
    run a forward pass, and record the drop in confidence for the
    predicted class.  High drop  →  that region matters a lot.
    """

    def __init__(self, model, device=None,
                 patch_size=16, stride=8, occlusion_value=0.5):
        """
        patch_size     : side length of the grey square (pixels in model input space)
        stride         : how many pixels to move the patch each step
        occlusion_value: fill value (0–1); 0.5 ≈ mid-grey after normalisation
        """
        self.model = model
        self.device = device if device else config.DEVICE
        self.patch_size = patch_size
        self.stride = stride
        self.occlusion_value = occlusion_value

        self.model.to(self.device)
        self.model.eval()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _baseline_confidence(self, img_tensor, class_idx):
        """Return the model's confidence on the unoccluded image."""
        with torch.no_grad():
            out = self.model(img_tensor.unsqueeze(0).to(self.device))
            prob = torch.softmax(out, dim=1)[0, class_idx].item()
        return prob

    def _occluded_confidence(self, img_tensor, class_idx, row, col):
        """Return confidence after occluding the patch at (row, col)."""
        occluded = img_tensor.clone()
        r1 = row
        r2 = min(row + self.patch_size, img_tensor.shape[1])
        c1 = col
        c2 = min(col + self.patch_size, img_tensor.shape[2])
        occluded[:, r1:r2, c1:c2] = self.occlusion_value
        with torch.no_grad():
            out = self.model(occluded.unsqueeze(0).to(self.device))
            prob = torch.softmax(out, dim=1)[0, class_idx].item()
        return prob

    # ── main map computation ──────────────────────────────────────────────────

    def compute_sensitivity_map(self, img_tensor, class_idx=None):
        """
        Returns
        -------
        sensitivity_map : np.ndarray (H, W)  values in [0, 1]
        pred_class      : int
        baseline_conf   : float
        """
        img_tensor = img_tensor.to(self.device)

        # Get predicted class if not supplied
        with torch.no_grad():
            out = self.model(img_tensor.unsqueeze(0))
            if class_idx is None:
                class_idx = out.argmax(dim=1).item()

        _, H, W = img_tensor.shape
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        count_map       = np.zeros((H, W), dtype=np.float32)

        baseline_conf = self._baseline_confidence(img_tensor, class_idx)

        rows = range(0, H, self.stride)
        cols = range(0, W, self.stride)
        total = len(rows) * len(cols)

        for row in tqdm(rows, desc="Occlusion rows", leave=False):
            for col in cols:
                occ_conf = self._occluded_confidence(img_tensor, class_idx, row, col)
                drop = max(0.0, baseline_conf - occ_conf)   # how much confidence fell

                r1, r2 = row, min(row + self.patch_size, H)
                c1, c2 = col, min(col + self.patch_size, W)
                sensitivity_map[r1:r2, c1:c2] += drop
                count_map[r1:r2, c1:c2]       += 1.0

        # Average overlapping patches
        count_map = np.maximum(count_map, 1)
        sensitivity_map /= count_map

        # Normalise to [0, 1]
        s_min, s_max = sensitivity_map.min(), sensitivity_map.max()
        if s_max > s_min:
            sensitivity_map = (sensitivity_map - s_min) / (s_max - s_min)

        return sensitivity_map, class_idx, baseline_conf

    # ── overlay helper ────────────────────────────────────────────────────────

    def overlay_heatmap(self, heatmap, image_np, alpha=0.45):
        """Blend a heatmap (H,W in [0,1]) onto an RGB image (H,W,3)."""
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        img = image_np / 255.0 if image_np.max() > 1.0 else image_np
        overlaid = (1 - alpha) * img + alpha * heatmap_colored
        return np.clip(overlaid, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Denormalise helper (same as gradcam)
# ─────────────────────────────────────────────────────────────────────────────

def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Batch visualisation  (test-set mode — no image path needed)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_occlusion_batch(model, data_loader, model_name,
                               num_samples=4, device=None,
                               patch_size=16, stride=8):
    if device is None:
        device = config.DEVICE

    print("\n" + "="*80)
    print(f"OCCLUSION SENSITIVITY: {model_name}")
    print(f"Patch size: {patch_size}px  |  Stride: {stride}px  |  Device: {device}")
    print("="*80)

    occluder = OcclusionSensitivity(model, device=device,
                                    patch_size=patch_size, stride=stride)

    # Collect samples — skip random offset so different images appear each run
    import random
    images_list, labels_list = [], []

    total_batches = len(data_loader)
    skip_batches  = random.randint(0, max(0, total_batches - num_samples - 1))
    print(f"Skipping {skip_batches} batches to randomise sample selection...")

    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx < skip_batches:
            continue
        for i in range(min(images.shape[0], num_samples - len(images_list))):
            images_list.append(images[i])
            labels_list.append(labels[i].item())
        if len(images_list) >= num_samples:
            break

    if not images_list:
        print("No images found in data loader!")
        return

    print(f"Selected {len(images_list)} samples.")

    fig, axes = plt.subplots(num_samples, 4,
                             figsize=(18, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(len(images_list)):
        img_tensor  = images_list[idx]
        true_label  = labels_list[idx]

        print(f"\nSample {idx+1}/{len(images_list)} — true label: "
              f"{config.IDX_TO_CLASS.get(true_label, true_label)}")

        sens_map, pred_class, baseline_conf = occluder.compute_sensitivity_map(
            img_tensor.to(device))

        pred_label = config.IDX_TO_CLASS.get(pred_class, pred_class)
        image_np   = denormalize_image(img_tensor)   # (H,W,3) in [0,1]
        image_uint8 = np.uint8(image_np * 255)

        overlay = occluder.overlay_heatmap(sens_map, image_uint8)

        # Col 0 — original
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(
            f"Original\nTrue: {config.IDX_TO_CLASS.get(true_label, true_label)}",
            fontsize=10)
        axes[idx, 0].axis('off')

        # Col 1 — sensitivity heatmap
        im = axes[idx, 1].imshow(sens_map, cmap='jet', vmin=0, vmax=1)
        axes[idx, 1].set_title(
            f"Occlusion Sensitivity\nPred: {pred_label} ({baseline_conf:.2%})",
            fontsize=10)
        axes[idx, 1].axis('off')
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        # Col 2 — overlay
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Sensitivity Overlay", fontsize=10)
        axes[idx, 2].axis('off')

        # Col 3 — stats
        axes[idx, 3].axis('off')
        stats = (
            f"OCCLUSION STATS\n\n"
            f"Predicted : {pred_label}\n"
            f"Baseline  : {baseline_conf:.2%}\n\n"
            f"Patch size: {patch_size}px\n"
            f"Stride    : {stride}px\n\n"
            f"Map mean  : {sens_map.mean():.4f}\n"
            f"Map max   : {sens_map.max():.4f}\n\n"
            f"High-sens pixels\n"
            f"(>0.5)    : {(sens_map > 0.5).sum()}"
        )
        axes[idx, 3].text(0.05, 0.95, stats, fontsize=10,
                          family='monospace', va='top',
                          transform=axes[idx, 3].transAxes,
                          bbox=dict(boxstyle='round',
                                    facecolor='lightyellow', alpha=0.4))

        # Green border = correct, red = wrong
        color = 'green' if pred_class == true_label else 'red'
        for ax in axes[idx, :3]:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    plt.suptitle(f"Occlusion Sensitivity — {model_name}",
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    occ_dir   = os.path.join(config.XAI_OUTPUT_DIR, "occlusion")
    os.makedirs(occ_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(occ_dir, f"{model_name}_occlusion_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nOK Occlusion visualizations saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-image visualisation  (like shap single-image mode)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_occlusion_single(model, image_path, model_name="ResNet18",
                                device=None, patch_size=16, stride=8,
                                save_path=None):
    if device is None:
        device = config.DEVICE

    transform   = get_transforms(augment=False)
    img_pil     = Image.open(image_path).convert('RGB')
    original_np = np.array(img_pil)
    img_tensor  = transform(img_pil).to(device)

    occluder = OcclusionSensitivity(model, device=device,
                                    patch_size=patch_size, stride=stride)
    print(f"Computing occlusion map for: {os.path.basename(image_path)}")
    sens_map, pred_class, baseline_conf = occluder.compute_sensitivity_map(
        img_tensor)

    pred_label = config.IDX_TO_CLASS.get(pred_class, pred_class)
    overlay    = occluder.overlay_heatmap(sens_map, original_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(sens_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(
        f"Occlusion Sensitivity\nPred: {pred_label} ({baseline_conf:.2%})",
        fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Sensitivity Overlay", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].axis('off')
    stats = (
        f"OCCLUSION STATS\n\n"
        f"Predicted : {pred_label}\n"
        f"Confidence: {baseline_conf:.2%}\n\n"
        f"Patch size: {patch_size}px\n"
        f"Stride    : {stride}px\n\n"
        f"Map mean  : {sens_map.mean():.4f}\n"
        f"Map max   : {sens_map.max():.4f}\n\n"
        f"High-sens pixels\n"
        f"(>0.5)    : {(sens_map > 0.5).sum()}\n\n"
        f"Interpretation:\n"
        f"Red areas = model relies\n"
        f"heavily on those pixels.\n"
        f"Covering them drops\n"
        f"confidence the most."
    )
    axes[3].text(0.05, 0.95, stats, fontsize=10,
                 family='monospace', va='top',
                 transform=axes[3].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))

    plt.suptitle(f"Occlusion Sensitivity — {model_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if not save_path:
        img_name  = os.path.basename(image_path).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        occ_dir   = os.path.join(config.XAI_OUTPUT_DIR, "occlusion")
        os.makedirs(occ_dir, exist_ok=True)
        save_path = os.path.join(occ_dir, f"occlusion_{img_name}_{timestamp}.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK Saved to: {save_path}")

    print(f"\nANALYSIS RESULTS:")
    print(f"   Predicted Sign : {pred_label}")
    print(f"   Confidence     : {baseline_conf:.2%}")
    print(f"   Map mean       : {sens_map.mean():.4f}")
    print(f"   Map max        : {sens_map.max():.4f}")
    print(f"   High-sens px   : {(sens_map > 0.5).sum()}")

    return sens_map, pred_class, baseline_conf


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Occlusion Sensitivity XAI — batch or single-image mode")
    parser.add_argument("image_path", nargs='?', default=None,
                        help="Optional: path to a single image. "
                             "If omitted, runs on a batch from the test set.")
    parser.add_argument("--output",      help="Custom output path for the PNG")
    parser.add_argument("--patch",  type=int, default=16,
                        help="Occlusion patch size in pixels (default 16)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride between patches in pixels (default 8)")
    parser.add_argument("--samples", type=int, default=4,
                        help="Number of test-set samples in batch mode (default 4)")
    args = parser.parse_args()

    print("="*80)
    print("OCCLUSION SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Patch size: {args.patch}px  |  Stride: {args.stride}px")

    # Load model
    resnet_path = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)
    if not os.path.exists(resnet_path):
        print(f"\nModel not found at {resnet_path}")
        print("Please run train.py first.")
        return

    model = create_resnet18_model()
    checkpoint = torch.load(resnet_path, map_location=config.DEVICE,
                             weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(config.DEVICE)
    model.eval()
    print("OK Model loaded.")

    # ── single-image mode ────────────────────────────────────────────────────
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Image not found: {args.image_path}")
            return
        visualize_occlusion_single(
            model, args.image_path, model_name="ResNet18",
            device=config.DEVICE,
            patch_size=args.patch, stride=args.stride,
            save_path=args.output)

    # ── batch mode (test set) ────────────────────────────────────────────────
    else:
        print("\nNo image path given — running on test-set batch...")
        data_loader_obj = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
        train_paths, train_labels, test_paths, test_labels = \
            data_loader_obj.load_data()

        if not test_paths:
            print("No test data available!")
            return

        _, _, test_loader = create_data_loaders(
            train_paths, train_labels, test_paths, test_labels)

        visualize_occlusion_batch(
            model, test_loader, "ResNet18",
            num_samples=args.samples,
            device=config.DEVICE,
            patch_size=args.patch,
            stride=args.stride)

    print("\n" + "="*80)
    print("OCCLUSION SENSITIVITY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved in: "
          f"{os.path.join(config.XAI_OUTPUT_DIR, 'occlusion')}")


if __name__ == "__main__":
    main()