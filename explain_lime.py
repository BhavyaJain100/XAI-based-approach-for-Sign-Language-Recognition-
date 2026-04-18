"""
Fixed LIME Explanations
"""

import os
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import config
from data_loader import SimpleDataLoader, create_data_loaders, get_transforms
from model_architecture import create_resnet18_model


class LIMEExplainer:
    """LIME explainer for PyTorch image classification models"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else config.DEVICE
        self.model.eval()
        self.model.to(self.device)
        self.explainer = lime_image.LimeImageExplainer()
        
        # Transform for preprocessing
        self.transform = get_transforms(augment=False)
    
    def predict_fn(self, images):
        """Prediction function for LIME"""
        # Convert numpy images to tensors
        batch_tensors = []
        for img in images:
            # LIME provides images in [0, 1] range
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_tensor = self.transform(img_pil)
            batch_tensors.append(img_tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def explain_instance(self, image, top_labels=3, num_samples=300, num_features=6):
        """Generate LIME explanation for a single image"""
        # Ensure image is in correct format
        if image.max() > 1.0:
            image = image / 255.0
        
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=config.LIME_HIDE_COLOR,
            num_samples=num_samples,
            batch_size=config.BATCH_SIZE
        )
        
        return explanation
    
    def visualize_explanation(self, image, explanation, label, positive_only=True):
        """Visualize LIME explanation"""
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            num_features=config.LIME_NUM_FEATURES,
            hide_rest=False
        )
        
        # Create visualization with boundaries
        img_boundary = mark_boundaries(temp, mask)
        
        return img_boundary, mask


def visualize_lime_batch(model, data_loader, model_name, num_samples=2, device=None):
    """Visualize LIME explanations for multiple samples"""
    if device is None:
        device = config.DEVICE
    
    print("\n" + "="*80)
    print(f"LIME VISUALIZATION: {model_name}")
    print("="*80)
    
    # Initialize LIME explainer
    lime_explainer = LIMEExplainer(model, device)
    
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
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(len(images_list)):
        print(f"\nProcessing sample {idx + 1}/{num_samples}...")
        
        # Get image and denormalize for LIME
        image_tensor = images_list[idx]
        true_label = labels_list[idx]
        
        # Denormalize image for LIME - ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image_for_lime = image_tensor * std + mean
        image_for_lime = torch.clamp(image_for_lime, 0, 1)
        image_np = image_for_lime.permute(1, 2, 0).cpu().numpy()
        
        # Get model prediction
        with torch.no_grad():
            img_input = image_tensor.unsqueeze(0).to(device)
            output = model(img_input)
            pred_probs = torch.softmax(output, dim=1)[0]
            pred_class = pred_probs.argmax().item()
            pred_conf = pred_probs[pred_class].item()
        
        # Generate LIME explanation
        print(f"  Generating LIME explanation...")
        explanation = lime_explainer.explain_instance(
            image_np, 
            top_labels=1,
            num_samples=config.LIME_NUM_SAMPLES,
            num_features=config.LIME_NUM_FEATURES
        )
        
        # Visualize for predicted class
        img_pos, mask_pos = lime_explainer.visualize_explanation(
            image_np, explanation, pred_class, positive_only=True
        )
        img_both, mask_both = lime_explainer.visualize_explanation(
            image_np, explanation, pred_class, positive_only=False
        )
        
        # Plot original image
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(
            f"Original\nTrue: {config.IDX_TO_CLASS.get(true_label, true_label)}",
            fontsize=10
        )
        axes[idx, 0].axis('off')
        
        # Plot prediction
        axes[idx, 1].imshow(image_np)
        axes[idx, 1].set_title(
            f"Prediction\n{config.IDX_TO_CLASS.get(pred_class, pred_class)} ({pred_conf:.2%})",
            fontsize=10
        )
        axes[idx, 1].axis('off')
        
        # Plot positive contributions
        axes[idx, 2].imshow(img_pos)
        axes[idx, 2].set_title(
            "LIME: Positive Features\n(Supporting prediction)",
            fontsize=10
        )
        axes[idx, 2].axis('off')
        
        # Plot all contributions
        axes[idx, 3].imshow(img_both)
        axes[idx, 3].set_title(
            "LIME: All Features\n(Positive + Negative)",
            fontsize=10
        )
        axes[idx, 3].axis('off')
        
        # Color code by correctness
        color = 'green' if pred_class == true_label else 'red'
        for ax in axes[idx, :]:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    plt.suptitle(f"LIME Explanations - {model_name}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    lime_dir = os.path.join(config.XAI_OUTPUT_DIR, "lime")
    os.makedirs(lime_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(lime_dir, f"{model_name}_lime_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ LIME visualizations saved to: {save_path}")


def main():
    """Main function for LIME explanations"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LIME — batch or single-image mode")
    parser.add_argument("image_path", nargs='?', default=None,
                        help="Optional: path to a single image. "
                             "If omitted, runs on a batch from the test set.")
    parser.add_argument("--samples", type=int, default=2,
                        help="Number of test-set samples in batch mode (default 2)")
    args = parser.parse_args()

    print("="*80)
    print("LIME EXPLAINABILITY ANALYSIS")
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
            from data_loader import get_transforms
            from PIL import Image as PILImage
            import torch as _torch

            transform  = get_transforms(augment=False)
            img_pil    = PILImage.open(args.image_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0)

            true_label_str = os.path.basename(os.path.dirname(args.image_path))
            class_to_idx   = {v: k for k, v in config.IDX_TO_CLASS.items()}
            true_label_idx = class_to_idx.get(true_label_str, 0)
            fake_loader    = [(img_tensor, _torch.tensor([true_label_idx]))]

            print(f"\nSingle-image mode: {args.image_path}")
            visualize_lime_batch(resnet_model, fake_loader,
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
            print("Analyzing ResNet18 Model with LIME")
            print("="*80)
            visualize_lime_batch(resnet_model, test_loader,
                                 "ResNet18", num_samples=args.samples)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("LIME ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved in: {os.path.join(config.XAI_OUTPUT_DIR, 'lime')}")


if __name__ == "__main__":
    main()