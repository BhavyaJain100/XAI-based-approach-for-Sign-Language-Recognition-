"""
Prediction script with comprehensive XAI
FIXED VERSION
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import config
from model_architecture import create_resnet18_model
from data_loader import get_simple_transforms
import torchvision.transforms as T

# Import XAI methods
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class XAIPredictor:
    """Complete predictor with all XAI methods"""
    
    def __init__(self):
        self.device = config.DEVICE
        self.model_type = 'resnet18'
        
        # Load ResNet18
        self.model = create_resnet18_model()
        model_path = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.val_acc = checkpoint.get('val_acc', 0)
        else:
            self.model.load_state_dict(checkpoint)
            self.val_acc = 0
        
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_simple_transforms(augment=False)
        
        print(f"OK Loaded ResNet18 model")
        if self.val_acc > 0:
            print(f"OK Validation accuracy: {self.val_acc:.2f}%")
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        original_img = img.copy()
        
        # Apply transforms
        img_tensor = self.transform(img)
        
        return img_tensor, np.array(original_img)
    
    def predict(self, img_tensor):
        """Make prediction with confidence scores"""
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            # Get top-5 predictions
            topk_probs, topk_indices = torch.topk(probs, 5)
            
            # Convert to lists
            topk_probs = topk_probs.cpu().numpy()
            topk_indices = topk_indices.cpu().numpy()
            topk_labels = [config.IDX_TO_CLASS[idx] for idx in topk_indices]
            
            predicted_label = topk_labels[0]
            confidence = float(topk_probs[0])
            
            return {
                'predicted': predicted_label,
                'confidence': confidence,
                'top5_labels': topk_labels,
                'top5_probs': topk_probs.tolist(),
                'top5_indices': topk_indices.tolist(),
                'raw_output': outputs,
                'probabilities': probs.cpu().numpy()
            }
    
    def explain_gradcam(self, img_tensor, target_class=None):
        """Generate Grad-CAM explanation"""
        from explain_gradcam import GradCAM
        target_layer = self.model.resnet.layer4[-1]
        grad_cam = GradCAM(self.model, target_layer)
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        cam, pred_class, _ = grad_cam.generate_cam(img_batch, target_class)
        return cam, pred_class
    
    def explain_lime(self, original_image, img_tensor, target_class=None):
        """Generate LIME explanation"""
        from explain_lime import LIMEExplainer
        
        # Initialize LIME
        lime_explainer = LIMEExplainer(self.model, self.device)
        
        # Prepare image for LIME
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
        
        # Generate explanation
        explanation = lime_explainer.explain_instance(
            original_image,
            top_labels=1,
            num_samples=300,
            num_features=6
        )
        
        # Get prediction
        pred_result = self.predict(img_tensor)
        target_idx = target_class if target_class else pred_result['top5_indices'][0]
        
        # Get visualization
        img_pos, mask_pos = lime_explainer.visualize_explanation(
            original_image, explanation, target_idx, positive_only=True
        )
        
        return img_pos, explanation, pred_result
    
    def explain_shap_simple(self, original_image, img_tensor, n_steps=50):
        """SHAP via Integrated Gradients — much richer maps than raw gradients."""
        img  = img_tensor.to(self.device).unsqueeze(0)
        base = torch.zeros_like(img)

        with torch.no_grad():
            out        = self.model(img)
            pred_class = out.argmax(dim=1).item()

        alphas      = torch.linspace(0, 1, n_steps, device=self.device)
        interp_imgs = (base + alphas[:, None, None, None] *
                       (img - base)).requires_grad_(True)

        scores = self.model(
            interp_imgs.view(-1, *img.shape[1:]))[:, pred_class].sum()
        self.model.zero_grad()
        scores.backward()

        avg_grads  = interp_imgs.grad.mean(dim=0)
        int_grads  = avg_grads * (img[0] - base[0])
        saliency   = int_grads.abs().mean(dim=0).cpu().detach().numpy()

        mn, mx = saliency.min(), saliency.max()
        if mx > mn:
            saliency = (saliency - mn) / (mx - mn)
        return saliency, pred_class

    def explain_occlusion(self, img_tensor, patch_size=16, stride=8):
        """Occlusion Sensitivity explanation"""
        from explain_occlusion import OcclusionSensitivity
        occluder = OcclusionSensitivity(self.model, device=self.device,
                                        patch_size=patch_size, stride=stride)
        sens_map, pred_class, baseline_conf = occluder.compute_sensitivity_map(
            img_tensor.to(self.device))
        return sens_map, pred_class, baseline_conf
    
    def visualize_all(self, original_image, img_tensor, save_path=None):
        """Create comprehensive visualization with all XAI methods"""
        print("\nGenerating comprehensive XAI analysis...")

        print("  Making prediction...")
        pred_result = self.predict(img_tensor)

        print("  Generating Grad-CAM...")
        gradcam_map, gradcam_class = self.explain_gradcam(img_tensor)

        print("  Generating LIME explanation...")
        lime_image, lime_expl, _ = self.explain_lime(original_image, img_tensor)

        print("  Generating SHAP importance map...")
        shap_map, shap_class = self.explain_shap_simple(original_image, img_tensor)

        print("  Generating Occlusion Sensitivity map...")
        occ_map, occ_class, occ_conf = self.explain_occlusion(img_tensor)

        # Overlays
        from explain_gradcam import GradCAM
        from explain_occlusion import OcclusionSensitivity
        import cv2
        import matplotlib.cm as cm

        target_layer    = self.model.resnet.layer4[-1]
        grad_cam_obj    = GradCAM(self.model, target_layer)
        gradcam_overlay = grad_cam_obj.overlay_heatmap(gradcam_map, original_image / 255.0)
        grad_cam_obj.remove_hooks()

        shap_colored  = cm.hot(shap_map)[:, :, :3]
        shap_resized  = cv2.resize(shap_colored, (original_image.shape[1], original_image.shape[0]))
        shap_overlay  = np.clip(0.5 * (original_image / 255.0) + 0.5 * shap_resized, 0, 1)

        occluder    = OcclusionSensitivity(self.model, device=self.device)
        occ_overlay = occluder.overlay_heatmap(occ_map, original_image)

        # 3×4 grid
        # Row 0: original | prediction text | gradcam heatmap | gradcam overlay
        # Row 1: lime     | shap heatmap    | shap overlay    | summary
        # Row 2: occlusion heatmap | occlusion overlay | interpretation (spans 2)
        fig = plt.figure(figsize=(20, 15))
        gs  = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

        # ── Row 0 ────────────────────────────────────────────────────────────
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.imshow(original_image)
        ax00.set_title("Original Image", fontsize=11, fontweight='bold')
        ax00.axis('off')

        ax01 = fig.add_subplot(gs[0, 1])
        ax01.axis('off')
        ty = 0.97
        ax01.text(0.05, ty, "PREDICTION", fontsize=12, fontweight='bold', transform=ax01.transAxes)
        ty -= 0.13
        ax01.text(0.05, ty, f"Sign: {pred_result['predicted']}", fontsize=11, transform=ax01.transAxes)
        ty -= 0.11
        ax01.text(0.05, ty, f"Confidence: {pred_result['confidence']:.2%}", fontsize=11, transform=ax01.transAxes)
        ty -= 0.14
        ax01.text(0.05, ty, "Top-5:", fontsize=10, fontweight='bold', transform=ax01.transAxes)
        ty -= 0.10
        for i, (label, prob) in enumerate(zip(pred_result['top5_labels'], pred_result['top5_probs'])):
            ax01.text(0.05, ty - i * 0.09, f"{i+1}. {label}: {prob:.2%}", fontsize=9, transform=ax01.transAxes)

        ax02 = fig.add_subplot(gs[0, 2])
        im_gc = ax02.imshow(gradcam_map, cmap='jet', vmin=0, vmax=1)
        ax02.set_title("Grad-CAM Heatmap", fontsize=11, fontweight='bold')
        ax02.axis('off')
        plt.colorbar(im_gc, ax=ax02, fraction=0.046, pad=0.04)

        ax03 = fig.add_subplot(gs[0, 3])
        ax03.imshow(gradcam_overlay)
        ax03.set_title("Grad-CAM Overlay", fontsize=11, fontweight='bold')
        ax03.axis('off')

        # ── Row 1 ────────────────────────────────────────────────────────────
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.imshow(lime_image)
        ax10.set_title("LIME Explanation", fontsize=11, fontweight='bold')
        ax10.axis('off')

        ax11 = fig.add_subplot(gs[1, 1])
        im_sh = ax11.imshow(shap_map, cmap='hot')
        ax11.set_title("SHAP Gradient Importance", fontsize=11, fontweight='bold')
        ax11.axis('off')
        plt.colorbar(im_sh, ax=ax11, fraction=0.046, pad=0.04)

        ax12 = fig.add_subplot(gs[1, 2])
        ax12.imshow(shap_overlay)
        ax12.set_title("SHAP Overlay", fontsize=11, fontweight='bold')
        ax12.axis('off')

        ax13 = fig.add_subplot(gs[1, 3])
        ax13.axis('off')
        summary = (
            f"XAI SUMMARY\n\n"
            f"Grad-CAM:\n  Attention {'concentrated' if gradcam_map.max() > 0.8 else 'distributed'}\n\n"
            f"LIME:\n  Top superpixels highlighted\n\n"
            f"SHAP:\n  Mean: {shap_map.mean():.4f}\n  Max:  {shap_map.max():.4f}\n\n"
            f"Occlusion:\n  Baseline: {occ_conf:.2%}\n  High-sens px: {(occ_map > 0.5).sum()}\n\n"
            f"Confidence:\n  {pred_result['confidence']:.2%}\n"
            f"  ({'High' if pred_result['confidence'] > 0.8 else 'Moderate' if pred_result['confidence'] > 0.5 else 'Low'})"
        )
        ax13.text(0.05, 0.97, summary, fontsize=9, family='monospace',
                  va='top', transform=ax13.transAxes,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # ── Row 2 : Occlusion ────────────────────────────────────────────────
        ax20 = fig.add_subplot(gs[2, 0])
        im_oc = ax20.imshow(occ_map, cmap='jet', vmin=0, vmax=1)
        ax20.set_title("Occlusion Sensitivity Heatmap", fontsize=11, fontweight='bold')
        ax20.axis('off')
        plt.colorbar(im_oc, ax=ax20, fraction=0.046, pad=0.04)

        ax21 = fig.add_subplot(gs[2, 1])
        ax21.imshow(occ_overlay)
        ax21.set_title("Occlusion Sensitivity Overlay", fontsize=11, fontweight='bold')
        ax21.axis('off')

        ax22 = fig.add_subplot(gs[2, 2:])
        ax22.axis('off')
        occ_text = (
            "OCCLUSION SENSITIVITY INTERPRETATION\n\n"
            "Red/hot regions = areas where covering with a grey patch\n"
            "caused the biggest confidence drop.\n"
            "These are the pixels the model relies on most.\n\n"
            f"Baseline confidence : {occ_conf:.2%}\n"
            f"Patch size          : 16px  |  Stride: 8px\n"
            f"High-sensitivity px : {(occ_map > 0.5).sum()} "
            f"({100*(occ_map > 0.5).mean():.1f}% of image)"
        )
        ax22.text(0.05, 0.85, occ_text, fontsize=10, family='monospace',
                  va='top', transform=ax22.transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))

        plt.suptitle("ISL Recognition — Grad-CAM + LIME + SHAP + Occlusion Sensitivity",
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"OK Comprehensive analysis saved to: {save_path}")

        plt.show(block=False)

        return pred_result, {
            'gradcam':   gradcam_map,
            'lime':      lime_image,
            'shap':      shap_map,
            'occlusion': occ_map
        }


def main():
    """Main function for XAI prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ISL Recognition with XAI")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--output", help="Output path for visualization")
    
    args = parser.parse_args()
    
    print("="*80)
    print("INDIAN SIGN LANGUAGE RECOGNITION WITH XAI")
    print("="*80)
    
    try:
        predictor = XAIPredictor()
        img_tensor, original_image = predictor.load_image(args.image_path)
        
        if args.output:
            output_path = args.output
        else:
            img_name = os.path.basename(args.image_path).split('.')[0]
            output_dir = os.path.join(config.RESULTS_DIR, "predictions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"resnet18_{img_name}_xai.png")
        
        pred_result, explanations = predictor.visualize_all(
            original_image, img_tensor, output_path
        )
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        print(f"\nPREDICTION SUMMARY:")
        print(f"   Model     : ResNet18")
        print(f"   Image     : {args.image_path}")
        print(f"   Predicted : {pred_result['predicted']}")
        print(f"   Confidence: {pred_result['confidence']:.2%}")
        
        print(f"\nTOP-5 PREDICTIONS:")
        for i, (label, prob) in enumerate(zip(pred_result['top5_labels'],
                                              pred_result['top5_probs']), 1):
            print(f"   {i}. {label}: {prob:.2%}")
        
        print(f"\nXAI METHODS USED:")
        print("   OK Grad-CAM          : Visual attention heatmap + overlay")
        print("   OK LIME              : Superpixel feature importance")
        print("   OK SHAP (gradient)   : Pixel-level gradient importance + overlay")
        
        print(f"\nOutput saved to: {output_path}")
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()