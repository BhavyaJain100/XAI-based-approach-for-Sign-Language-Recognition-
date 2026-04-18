"""
Fixed Grad-CAM - Addresses Device Mismatch Issues
"""

import os
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import config
from data_loader import SimpleDataLoader, create_data_loaders, get_transforms
from model_architecture import create_resnet18_model


class GradCAM:
    """Grad-CAM implementation with proper device handling"""
    
    def __init__(self, model, target_layer, device=None):
        self.model = model
        self.device = device if device else config.DEVICE
        self.model.to(self.device)
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        
        # Register hooks — do NOT detach activations so gradients flow
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient))
    
    def _save_activation(self, module, input, output):
        self.activations = output  # keep in graph for backward

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # detach only after backward

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Forward pass — keep grad enabled
        self.model.zero_grad()
        with torch.enable_grad():
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            # Backward on the target class score
            score = output[0, class_idx]
            score.backward()

        # gradients: (C, H, W)  activations: (1, C, H, W)
        gradients  = self.gradients[0]               # (C, H, W)
        activations = self.activations[0].detach()   # (C, H, W)

        # Global average pool gradients
        weights = gradients.mean(dim=(1, 2))         # (C,)

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:],
                          dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # Fallback: use raw activations mean if gradients vanished
            cam = activations.mean(dim=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.cpu().numpy(), class_idx, output
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """Overlay heatmap on image"""
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        # Ensure image is in correct range
        if image.max() <= 1.0:
            image_display = image
        else:
            image_display = image / 255.0
        
        # Overlay
        overlaid = (1 - alpha) * image_display + alpha * heatmap_colored
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid


def denormalize_image(tensor):
    """Denormalize image tensor to [0, 1] range"""
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    image = tensor.permute(1, 2, 0).cpu().numpy()
    
    return image


def visualize_gradcam_batch(model, data_loader, model_name, num_samples=4, device=None):
    """Visualize Grad-CAM for multiple samples"""
    if device is None:
        device = config.DEVICE
    
    print("\n" + "="*80)
    print(f"GRAD-CAM VISUALIZATION: {model_name}")
    print(f"Using device: {device}")
    print("="*80)
    
    model.eval()
    model.to(device)
    
    # Get target layer — use layer4[-1] which has richest spatial features
    target_layer = model.resnet.layer4[-1]
    print(f"Using target layer: layer4[-1] ({target_layer.__class__.__name__})")
    
    # Collect samples — skip a random offset so different images appear each run
    import random
    images_list = []
    labels_list = []

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

    if len(images_list) == 0:
        print("No images found in data loader!")
        return

    print(f"Selected {len(images_list)} samples.")

    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(len(images_list)):
        image_tensor = images_list[idx]
        true_label   = labels_list[idx]
        
        # Create a fresh GradCAM per sample to avoid stale hook state
        grad_cam = GradCAM(model, target_layer, device=device)
        cam, pred_class, output = grad_cam.generate_cam(
            image_tensor.unsqueeze(0).to(device))
        grad_cam.remove_hooks()

        pred_conf = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Denormalize image
        image    = denormalize_image(images_list[idx])
        overlaid = grad_cam.overlay_heatmap(cam, image, alpha=config.GRADCAM_ALPHA)
        
        # Col 0 — original
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(
            f"Original\nTrue: {config.IDX_TO_CLASS.get(true_label, true_label)}",
            fontsize=10)
        axes[idx, 0].axis('off')
        
        # Col 1 — heatmap
        axes[idx, 1].imshow(cam, cmap='jet', vmin=0, vmax=1)
        axes[idx, 1].set_title(
            f"Grad-CAM Heatmap\nPred: {config.IDX_TO_CLASS.get(pred_class, pred_class)} ({pred_conf:.2%})",
            fontsize=10)
        axes[idx, 1].axis('off')
        
        # Col 2 — overlay
        axes[idx, 2].imshow(overlaid)
        axes[idx, 2].set_title(f"Overlay\nConfidence: {pred_conf:.2%}", fontsize=10)
        axes[idx, 2].axis('off')
        
        # Green = correct, red = wrong
        color = 'green' if pred_class == true_label else 'red'
        for ax in axes[idx, :]:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    plt.suptitle(f"Grad-CAM Explanations - {model_name}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    gradcam_dir = os.path.join(config.XAI_OUTPUT_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(gradcam_dir, f"{model_name}_gradcam_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grad-CAM visualizations saved to: {save_path}")


def main():
    """Main function for Grad-CAM explanations"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Grad-CAM — batch or single-image mode")
    parser.add_argument("image_path", nargs='?', default=None,
                        help="Optional: path to a single image. "
                             "If omitted, runs on a batch from the test set.")
    parser.add_argument("--samples", type=int, default=4,
                        help="Number of test-set samples in batch mode (default 4)")
    args = parser.parse_args()

    print("="*80)
    print("GRAD-CAM EXPLAINABILITY ANALYSIS")
    print("="*80)
    print(f"Device: {config.DEVICE}")

    # Load model
    resnet_path = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)
    if not os.path.exists(resnet_path):
        print(f"\nModel not found at {resnet_path}. Please run train.py first.")
        return

    resnet_model = create_resnet18_model()
    checkpoint = torch.load(resnet_path, map_location=config.DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        resnet_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        resnet_model.load_state_dict(checkpoint)

    try:
        # ── single-image mode ────────────────────────────────────────────────
        if args.image_path:
            if not os.path.exists(args.image_path):
                print(f"Image not found: {args.image_path}")
                return
            # Build a one-image loader and run batch function with num_samples=1
            from data_loader import get_transforms
            from PIL import Image as PILImage
            import torch as _torch

            transform  = get_transforms(augment=False)
            img_pil    = PILImage.open(args.image_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0)

            # Wrap in a simple iterable so visualize_gradcam_batch works
            true_label_str = os.path.basename(os.path.dirname(args.image_path))
            class_to_idx   = {v: k for k, v in config.IDX_TO_CLASS.items()}
            true_label_idx = class_to_idx.get(true_label_str, 0)
            fake_loader    = [(img_tensor, _torch.tensor([true_label_idx]))]

            print(f"\nSingle-image mode: {args.image_path}")
            visualize_gradcam_batch(resnet_model, fake_loader,
                                    "ResNet18", num_samples=1)

        # ── batch mode ───────────────────────────────────────────────────────
        else:
            print("\nLoading test data...")
            data_loader_obj = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
            train_paths, train_labels, test_paths, test_labels = \
                data_loader_obj.load_data()

            if not test_paths:
                print("No test data available!")
                return

            _, _, test_loader = create_data_loaders(
                train_paths, train_labels, test_paths, test_labels)

            print(f"Test samples: {len(test_paths)}")
            print("\n" + "="*80)
            print("Analyzing ResNet18 Model with Grad-CAM")
            print("="*80)
            visualize_gradcam_batch(resnet_model, test_loader,
                                    "ResNet18", num_samples=args.samples)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("GRAD-CAM ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved in: {os.path.join(config.XAI_OUTPUT_DIR, 'gradcam')}")


if __name__ == "__main__":
    main()