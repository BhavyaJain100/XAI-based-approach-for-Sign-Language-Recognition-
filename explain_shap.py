"""
SHAP / Gradient Importance Explainability

Works in two modes — consistent with gradcam, lime, and occlusion:
  1. Batch mode  : no image argument — randomly picks samples from the test set
  2. Single-image: pass an image path on the command line

Results are saved in results/xai_explanations/shap/ with timestamps.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from datetime import datetime

import config
from model_architecture import create_resnet18_model
from data_loader import (SimpleDataLoader, create_data_loaders,
                         get_transforms, get_simple_transforms)


# ─────────────────────────────────────────────────────────────────────────────
# Core SHAP class
# ─────────────────────────────────────────────────────────────────────────────

class SimpleSHAP:
    """
    SHAP via Integrated Gradients.
    Averages gradients along 50 interpolation steps from a black
    baseline to the input — produces clear, spatially meaningful maps
    instead of the near-zero raw-gradient maps.
    """

    def __init__(self, model, device=None):
        self.device = device if device else config.DEVICE
        self.model  = model
        self.model.to(self.device)
        self.model.eval()

    def compute_importance(self, img_tensor, n_steps=50):
        """
        Returns
        -------
        importance_map : np.ndarray (H, W)  in [0, 1]
        pred_class     : int
        confidence     : float
        """
        img    = img_tensor.to(self.device).unsqueeze(0)   # (1,C,H,W)
        base   = torch.zeros_like(img)

        # Get predicted class first (no grad needed)
        with torch.no_grad():
            out        = self.model(img)
            probs      = torch.softmax(out, dim=1)[0]
            pred_class = out.argmax(dim=1).item()
            confidence = probs[pred_class].item()

        # Integrated Gradients — n_steps interpolations
        alphas      = torch.linspace(0, 1, n_steps, device=self.device)
        interp_imgs = base + alphas[:, None, None, None] * (img - base)
        interp_imgs = interp_imgs.requires_grad_(True)

        scores = self.model(interp_imgs.view(-1, *img.shape[1:]))[:,
                                                                   pred_class].sum()
        self.model.zero_grad()
        scores.backward()

        avg_grads      = interp_imgs.grad.mean(dim=0)          # (C,H,W)
        int_grads      = avg_grads * (img[0] - base[0])        # (C,H,W)
        importance_map = int_grads.abs().mean(dim=0).cpu().detach().numpy()

        mn, mx = importance_map.min(), importance_map.max()
        if mx > mn:
            importance_map = (importance_map - mn) / (mx - mn)

        return importance_map, pred_class, confidence

    @staticmethod
    def overlay(importance_map, image_np):
        """Blend hot-colormap importance onto an RGB image."""
        import cv2
        imp_resized = cv2.resize(importance_map,
                                 (image_np.shape[1], image_np.shape[0]))
        heatmap = cm.hot(imp_resized)[:, :, :3]
        img     = image_np / 255.0 if image_np.max() > 1.0 else image_np
        blended = np.clip(0.5 * img + 0.5 * heatmap, 0, 1)
        return blended


# ─────────────────────────────────────────────────────────────────────────────
# Denormalize helper
# ─────────────────────────────────────────────────────────────────────────────

def _denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t    = torch.clamp(tensor * std + mean, 0, 1)
    return t.permute(1, 2, 0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode  (test-set, random samples — like gradcam / lime / occlusion)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_shap_batch(model, data_loader, model_name,
                          num_samples=4, device=None):
    if device is None:
        device = config.DEVICE

    print("\n" + "="*80)
    print(f"SHAP VISUALIZATION (batch): {model_name}")
    print(f"Device: {device}")
    print("="*80)

    explainer = SimpleSHAP(model, device=device)

    # Collect samples — skip random offset so different images appear each run
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

    fig, axes = plt.subplots(num_samples, 4, figsize=(18, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, (img_tensor, true_label) in enumerate(zip(images_list, labels_list)):
        imp_map, pred_class, confidence = explainer.compute_importance(img_tensor)

        image_np    = _denormalize(img_tensor)
        image_uint8 = np.uint8(image_np * 255)
        overlay_img = SimpleSHAP.overlay(imp_map, image_uint8)

        pred_label = config.IDX_TO_CLASS.get(pred_class, pred_class)
        true_str   = config.IDX_TO_CLASS.get(true_label, true_label)

        # Col 0 — original
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f"Original\nTrue: {true_str}", fontsize=10)
        axes[idx, 0].axis('off')

        # Col 1 — importance heatmap
        im = axes[idx, 1].imshow(imp_map, cmap='hot')
        axes[idx, 1].set_title(
            f"SHAP Importance\nPred: {pred_label} ({confidence:.2%})",
            fontsize=10)
        axes[idx, 1].axis('off')
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        # Col 2 — overlay
        axes[idx, 2].imshow(overlay_img)
        axes[idx, 2].set_title("Importance Overlay", fontsize=10)
        axes[idx, 2].axis('off')

        # Col 3 — stats
        axes[idx, 3].axis('off')
        stats = (
            f"SHAP STATS\n\n"
            f"Predicted : {pred_label}\n"
            f"Confidence: {confidence:.2%}\n\n"
            f"Mean imp  : {imp_map.mean():.4f}\n"
            f"Max imp   : {imp_map.max():.4f}\n\n"
            f"High-imp px\n"
            f"(>0.5)    : {(imp_map > 0.5).sum()}"
        )
        axes[idx, 3].text(0.05, 0.95, stats, fontsize=10,
                          family='monospace', va='top',
                          transform=axes[idx, 3].transAxes,
                          bbox=dict(boxstyle='round',
                                    facecolor='lightyellow', alpha=0.4))

        color = 'green' if pred_class == true_label else 'red'
        for ax in axes[idx, :3]:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    plt.suptitle(f"SHAP Gradient Importance — {model_name}",
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    shap_dir  = os.path.join(config.XAI_OUTPUT_DIR, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(shap_dir, f"{model_name}_shap_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nOK SHAP visualizations saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-image mode  (pass a path on the command line)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_shap_single(model, image_path, model_name="ResNet18",
                           device=None, save_path=None):
    if device is None:
        device = config.DEVICE

    transform   = get_simple_transforms(augment=False)
    img_pil     = Image.open(image_path).convert('RGB')
    original_np = np.array(img_pil)
    img_tensor  = transform(img_pil)

    explainer = SimpleSHAP(model, device=device)
    print(f"\nAnalyzing image: {image_path}")
    imp_map, pred_class, confidence = explainer.compute_importance(img_tensor)

    pred_label  = config.IDX_TO_CLASS.get(pred_class, pred_class)
    overlay_img = SimpleSHAP.overlay(imp_map, original_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(imp_map, cmap='hot')
    axes[1].set_title(
        f"Feature Importance\nPredicted: {pred_label} ({confidence:.2%})",
        fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_img)
    axes[2].set_title("Importance Overlay", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Feature Importance Analysis (SHAP)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if not save_path:
        img_name  = os.path.basename(image_path).split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shap_dir  = os.path.join(config.XAI_OUTPUT_DIR, "shap")
        os.makedirs(shap_dir, exist_ok=True)
        save_path = os.path.join(shap_dir, f"shap_{img_name}_{timestamp}.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK Saved to: {save_path}")

    print(f"\nANALYSIS RESULTS:")
    print(f"   Predicted Sign : {pred_label}")
    print(f"   Confidence     : {confidence:.2%}")
    print(f"   Mean importance: {imp_map.mean():.4f}")
    print(f"   Max importance : {imp_map.max():.4f}")
    threshold        = imp_map.mean() * 2
    important_pixels = int((imp_map > threshold).sum())
    total_pixels     = imp_map.size
    print(f"   Important px   : {important_pixels}/{total_pixels} "
          f"({important_pixels/total_pixels:.1%})")

    return imp_map, pred_class, confidence


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SHAP Gradient Importance — batch or single-image mode")
    parser.add_argument("image_path", nargs='?', default=None,
                        help="Optional: path to a single image. "
                             "If omitted, runs on a batch from the test set.")
    parser.add_argument("--model",   default=os.path.join(
                            config.MODEL_DIR, config.RESNET_MODEL_NAME),
                        help="Path to model checkpoint")
    parser.add_argument("--output",  help="Custom output path for the PNG")
    parser.add_argument("--samples", type=int, default=4,
                        help="Number of test-set samples in batch mode (default 4)")
    args = parser.parse_args()

    print("="*80)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    print(f"Device: {config.DEVICE}")

    if not os.path.exists(args.model):
        print(f"\nModel not found: {args.model}")
        print("Please run train.py first.")
        return

    model      = create_resnet18_model()
    checkpoint = torch.load(args.model, map_location=config.DEVICE,
                             weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(config.DEVICE)
    model.eval()
    print("OK Model loaded.")

    try:
        if args.image_path:
            # Single-image mode
            if not os.path.exists(args.image_path):
                print(f"Image not found: {args.image_path}")
                return
            visualize_shap_single(model, args.image_path,
                                   device=config.DEVICE,
                                   save_path=args.output)
        else:
            # Batch mode — randomly pick from test set
            print("\nNo image path given — running on test-set batch...")
            data_loader_obj = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
            train_paths, train_labels, test_paths, test_labels = \
                data_loader_obj.load_data()

            if not test_paths:
                print("No test data available!")
                return

            _, _, test_loader = create_data_loaders(
                train_paths, train_labels, test_paths, test_labels)

            visualize_shap_batch(model, test_loader, "ResNet18",
                                  num_samples=args.samples,
                                  device=config.DEVICE)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("SHAP ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved in: "
          f"{os.path.join(config.XAI_OUTPUT_DIR, 'shap')}")


if __name__ == "__main__":
    main()