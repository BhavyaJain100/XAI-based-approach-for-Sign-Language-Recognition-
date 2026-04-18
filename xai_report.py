"""
XAI Report Generator — single PDF for one or multiple images.

Structure:
  Page 1         : Cover / intro
  Pages 2-3      : Sample 1  (Page A = XAI visuals, Page B = analysis text)
  Pages 4-5      : Sample 2
  ...
Batch default: 4 images → 9-page PDF (1 cover + 2 pages × 4 samples)
Single image : 1 image  → 3-page PDF (1 cover + 2 pages × 1 sample)

SHAP uses Integrated Gradients (50 steps) for rich attribution maps.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from datetime import datetime

import config
from model_architecture import create_resnet18_model
from data_loader import (get_simple_transforms, SimpleDataLoader,
                         create_data_loaders)
from explain_gradcam import GradCAM, denormalize_image
from explain_lime import LIMEExplainer


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients — much better than raw gradients for SHAP maps
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(model, img_tensor, pred_idx, device, n_steps=50):
    """
    Average gradients along a linear path from a black baseline to the image.
    Returns an importance map in [0, 1] with clear spatial structure.
    """
    img   = img_tensor.unsqueeze(0).to(device)           # (1, C, H, W)
    base  = torch.zeros_like(img)                        # black baseline

    # Interpolate n_steps images between baseline and input
    alphas      = torch.linspace(0, 1, n_steps, device=device)
    interp_imgs = base + alphas[:, None, None, None] * (img - base)
    interp_imgs = interp_imgs.requires_grad_(True)

    # Forward all steps at once
    outputs = model(interp_imgs.view(-1, *img.shape[1:]))
    scores  = outputs[:, pred_idx].sum()

    model.zero_grad()
    scores.backward()

    # Average gradients × (input − baseline)
    avg_grads  = interp_imgs.grad.mean(dim=0)            # (C, H, W)
    int_grads  = avg_grads * (img[0] - base[0])         # (C, H, W)
    importance = int_grads.abs().mean(dim=0)             # (H, W)
    importance = importance.cpu().detach().numpy()

    mn, mx = importance.min(), importance.max()
    if mx > mn:
        importance = (importance - mn) / (mx - mn)
    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

class XAIReportGenerator:

    def __init__(self):
        self.device     = config.DEVICE
        self.models     = {}
        self.model_info = {}
        self.transform  = get_simple_transforms(augment=False)

        path = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        m    = create_resnet18_model()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            m.load_state_dict(ckpt['model_state_dict'])
            self.model_info['ResNet18'] = {
                'val_acc': ckpt.get('val_acc', 0),
                'epoch':   ckpt.get('epoch', 0)
            }
        else:
            m.load_state_dict(ckpt)
            self.model_info['ResNet18'] = {'val_acc': 0, 'epoch': 0}

        m.to(self.device)
        m.eval()
        self.models['ResNet18'] = m
        print(f"OK Model loaded — val_acc = "
              f"{self.model_info['ResNet18']['val_acc']:.2f}%")

    # ── helpers ───────────────────────────────────────────────────────────────

    def load_image(self, image_path):
        img          = Image.open(image_path).convert('RGB')
        original_img = np.array(img)
        img_tensor   = self.transform(img)
        return img_tensor, original_img

    def _shap_overlay(self, shap_map, original_img):
        """Blend hot-colormap SHAP map onto original image."""
        col     = cm.hot(shap_map)[:, :, :3]
        resized = cv2.resize(col, (original_img.shape[1], original_img.shape[0]))
        base    = original_img / 255.0
        return np.clip(0.5 * base + 0.5 * resized, 0, 1)

    # ── analysis ─────────────────────────────────────────────────────────────

    def analyze_image(self, img_tensor, original_img, model_name):
        model = self.models[model_name]

        # Prediction
        with torch.no_grad():
            batch   = img_tensor.unsqueeze(0).to(self.device)
            outputs = model(batch)
            probs   = torch.softmax(outputs, dim=1)[0]
            kp, ki  = torch.topk(probs, 5)
            pred = {
                'predicted_idx':   ki[0].item(),
                'predicted_label': config.IDX_TO_CLASS[ki[0].item()],
                'confidence':      kp[0].item(),
                'top5_labels':     [config.IDX_TO_CLASS[i.item()] for i in ki],
                'top5_probs':      kp.cpu().numpy()
            }

        image_norm = original_img / 255.0

        # Grad-CAM
        tl       = model.resnet.layer4[-1]
        gc       = GradCAM(model, tl)
        cam, _, _ = gc.generate_cam(batch)
        gc_overlay = gc.overlay_heatmap(cam, image_norm)
        gc.remove_hooks()

        # LIME
        lime_exp = LIMEExplainer(model, self.device)
        img_lime = image_norm if original_img.max() > 1.0 else original_img
        expl     = lime_exp.explain_instance(img_lime, top_labels=1,
                                             num_samples=200, num_features=6)
        lime_img, _ = lime_exp.visualize_explanation(
            img_lime, expl, pred['predicted_idx'], positive_only=True)

        # SHAP via Integrated Gradients (50 steps)
        shap_map    = integrated_gradients(model, img_tensor,
                                           pred['predicted_idx'],
                                           self.device, n_steps=50)
        shap_overlay = self._shap_overlay(shap_map, original_img)

        # Occlusion Sensitivity
        from explain_occlusion import OcclusionSensitivity
        occ = OcclusionSensitivity(model, device=self.device,
                                   patch_size=16, stride=8)
        occ_map, _, occ_conf = occ.compute_sensitivity_map(
            img_tensor.to(self.device))
        occ_overlay = occ.overlay_heatmap(occ_map, original_img)

        return {
            'prediction':  pred,
            'gradcam':     cam,
            'gradcam_ov':  gc_overlay,
            'lime':        lime_img,
            'shap':        shap_map,
            'shap_ov':     shap_overlay,
            'occlusion':   occ_map,
            'occ_ov':      occ_overlay,
            'occ_conf':    occ_conf,
        }

    # ── page builders ─────────────────────────────────────────────────────────

    def _cover_page(self, pdf, num_samples):
        fig = plt.figure(figsize=(8.5, 11))
        ax  = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.88,
                "XAI-Based Indian Sign Language\nRecognition Report",
                ha='center', va='center', fontsize=22, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.78, "Explainable AI Analysis",
                ha='center', va='center', fontsize=15, style='italic',
                transform=ax.transAxes)
        ax.text(0.5, 0.72,
                f"Generated: {datetime.now().strftime('%B %d, %Y   %H:%M')}",
                ha='center', va='center', fontsize=11, transform=ax.transAxes)

        y = 0.62
        ax.text(0.5, y, "Model Performance", ha='center', fontsize=13,
                fontweight='bold', transform=ax.transAxes)
        y -= 0.06
        for name, info in self.model_info.items():
            ax.text(0.5, y,
                    f"{name}:  val_acc = {info['val_acc']:.2f}%   "
                    f"(best epoch {info['epoch']})",
                    ha='center', fontsize=11, transform=ax.transAxes)
            y -= 0.05

        y -= 0.04
        ax.text(0.5, y, "Report Structure", ha='center', fontsize=13,
                fontweight='bold', transform=ax.transAxes)
        y -= 0.06
        n_pages = 1 + 2 * num_samples
        ax.text(0.5, y,
                f"This report analyses {num_samples} "
                f"image{'s' if num_samples > 1 else ''}  "
                f"({n_pages} pages total).\n"
                f"Each sample occupies 2 pages:\n"
                f"  Page A — XAI visualisations "
                f"(Grad-CAM, LIME, SHAP, Occlusion)\n"
                f"  Page B — Detailed analysis text and statistics",
                ha='center', va='top', fontsize=10,
                transform=ax.transAxes)

        y -= 0.20
        about = (
            "XAI Methods Used\n\n"
            "Grad-CAM (gradient-weighted class activation map)\n"
            "  Highlights which spatial regions the model focuses on\n"
            "  via gradient-weighted feature activation maps.\n\n"
            "LIME (local interpretable model-agnostic explanations)\n"
            "  Perturbs superpixels to find which image regions locally\n"
            "  support or oppose the prediction.\n\n"
            "SHAP — Integrated Gradients (50 steps)\n"
            "  Averages gradients along a path from a black baseline\n"
            "  to the input, giving pixel-level attribution maps.\n\n"
            "Occlusion Sensitivity (16x16 patch, stride 8)\n"
            "  Slides a grey patch over the image; records confidence\n"
            "  drop — larger drop = more important region.\n\n"
            "Colour key:\n"
            "  Red / hot  = high importance\n"
            "  Blue / cool = low importance\n"
            "  Green image border = correct prediction\n"
            "  Red image border   = incorrect prediction"
        )
        ax.text(0.08, y, about, va='top', fontsize=9,
                family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _visuals_page(self, pdf, image_path, original_img, analysis):
        """Page A — all 4 XAI methods for one sample (landscape)."""
        pred  = analysis['prediction']
        label = pred['predicted_label']
        conf  = pred['confidence']

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(
            f"XAI Visualisations  —  {os.path.basename(image_path)}"
            f"      Predicted: {label}  ({conf:.1%})",
            fontsize=12, fontweight='bold', y=0.98)

        # Layout: 3 rows x 5 cols
        # Row 0: [original] [gradcam heatmap] [    ] [shap heatmap] [occ heatmap]
        # Row 1: [original] [gradcam overlay] [lime] [shap overlay] [occ overlay]
        # Row 2: [original] [gradcam overlay] [lime] [shap overlay] [occ overlay]
        # Simpler: 2 rows, 5 cols; original spans rows
        gs = fig.add_gridspec(2, 5, hspace=0.38, wspace=0.25,
                              top=0.93, bottom=0.04,
                              left=0.03, right=0.97)

        # Original (spans both rows)
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.imshow(original_img)
        ax0.set_title("Original", fontsize=10, fontweight='bold')
        ax0.axis('off')

        # Grad-CAM heatmap (row 0) + overlay (row 1)
        ax_gc = fig.add_subplot(gs[0, 1])
        im1   = ax_gc.imshow(analysis['gradcam'], cmap='jet', vmin=0, vmax=1)
        ax_gc.set_title("Grad-CAM", fontsize=10, fontweight='bold')
        ax_gc.axis('off')
        plt.colorbar(im1, ax=ax_gc, fraction=0.046, pad=0.04)

        ax_gc_ov = fig.add_subplot(gs[1, 1])
        ax_gc_ov.imshow(analysis['gradcam_ov'])
        ax_gc_ov.set_title("Grad-CAM overlay", fontsize=9)
        ax_gc_ov.axis('off')

        # LIME (spans both rows — boundary image already shows full context)
        ax_lime = fig.add_subplot(gs[:, 2])
        ax_lime.imshow(analysis['lime'])
        ax_lime.set_title("LIME", fontsize=10, fontweight='bold')
        ax_lime.axis('off')

        # SHAP heatmap (row 0) + overlay (row 1)
        ax_sh = fig.add_subplot(gs[0, 3])
        im2   = ax_sh.imshow(analysis['shap'], cmap='hot', vmin=0, vmax=1)
        ax_sh.set_title("SHAP (Integrated Gradients)", fontsize=9,
                         fontweight='bold')
        ax_sh.axis('off')
        plt.colorbar(im2, ax=ax_sh, fraction=0.046, pad=0.04)

        ax_sh_ov = fig.add_subplot(gs[1, 3])
        ax_sh_ov.imshow(analysis['shap_ov'])
        ax_sh_ov.set_title("SHAP overlay", fontsize=9)
        ax_sh_ov.axis('off')

        # Occlusion heatmap (row 0) + overlay (row 1)
        ax_oc = fig.add_subplot(gs[0, 4])
        im3   = ax_oc.imshow(analysis['occlusion'], cmap='jet', vmin=0, vmax=1)
        ax_oc.set_title("Occlusion Sensitivity", fontsize=9,
                         fontweight='bold')
        ax_oc.axis('off')
        plt.colorbar(im3, ax=ax_oc, fraction=0.046, pad=0.04)

        ax_oc_ov = fig.add_subplot(gs[1, 4])
        ax_oc_ov.imshow(analysis['occ_ov'])
        ax_oc_ov.set_title("Occlusion overlay", fontsize=9)
        ax_oc_ov.axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _analysis_page(self, pdf, image_path, model_name, analysis):
        """Page B — prediction bar chart + statistics + interpretation text."""
        pred  = analysis['prediction']
        label = pred['predicted_label']
        conf  = pred['confidence']
        info  = self.model_info[model_name]

        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"Analysis  —  {os.path.basename(image_path)}",
                     fontsize=14, fontweight='bold', y=0.98)

        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3,
                              top=0.93, bottom=0.05,
                              left=0.08, right=0.95)

        # Top-left: horizontal bar chart of top-5
        ax_bar = fig.add_subplot(gs[0, 0])
        labels = pred['top5_labels']
        probs  = pred['top5_probs']
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(labels))]
        ax_bar.barh(labels[::-1], probs[::-1], color=colors[::-1])
        for j, (lbl, prob) in enumerate(zip(labels[::-1], probs[::-1])):
            ax_bar.text(prob + 0.01, j, f'{prob:.1%}',
                        va='center', fontsize=8)
        ax_bar.set_xlim(0, 1.1)
        ax_bar.set_xlabel("Confidence", fontsize=9)
        ax_bar.set_title("Top-5 Predictions", fontsize=11, fontweight='bold')
        ax_bar.tick_params(labelsize=9)

        # Top-right: stats panel
        ax_stats = fig.add_subplot(gs[0, 1])
        ax_stats.axis('off')
        certainty = ('Very High' if conf > 0.9
                     else 'High' if conf > 0.7
                     else 'Moderate' if conf > 0.5
                     else 'Low')
        stats = (
            f"PREDICTION\n"
            f"{'─'*30}\n"
            f"Sign        : {label}\n"
            f"Confidence  : {conf:.2%}\n"
            f"Certainty   : {certainty}\n\n"
            f"MODEL\n"
            f"{'─'*30}\n"
            f"Architecture: {model_name}\n"
            f"Val accuracy: {info['val_acc']:.2f}%\n"
            f"Best epoch  : {info['epoch']}\n\n"
            f"XAI STATS\n"
            f"{'─'*30}\n"
            f"Grad-CAM    : "
            f"{'concentrated' if analysis['gradcam'].max() > 0.8 else 'distributed'}\n"
            f"SHAP mean   : {float(analysis['shap'].mean()):.4f}\n"
            f"SHAP max    : {float(analysis['shap'].max()):.4f}\n"
            f"Occ baseline: {analysis['occ_conf']:.2%}\n"
            f"High-sens px: {int((analysis['occlusion'] > 0.5).sum())}"
        )
        ax_stats.text(0.02, 0.97, stats, va='top', fontsize=9,
                      family='monospace', transform=ax_stats.transAxes,
                      bbox=dict(boxstyle='round', facecolor='lightyellow',
                                alpha=0.4))

        # Bottom: full-width interpretation text
        ax_txt = fig.add_subplot(gs[1, :])
        ax_txt.axis('off')

        alts = ""
        for i in range(1, min(4, len(pred['top5_labels']))):
            tag = ('close competitor' if pred['top5_probs'][i] > 0.1
                   else 'low probability')
            alts += (f"  {i+1}. {pred['top5_labels'][i]} "
                     f"({pred['top5_probs'][i]:.1%}) - {tag}\n")

        body = (
            f"DETAILED XAI INTERPRETATION\n"
            f"{'─'*68}\n\n"
            f"Grad-CAM Analysis:\n"
            f"  The heatmap highlights which spatial regions were most important\n"
            f"  for predicting '{label}'. Red/yellow regions had the strongest\n"
            f"  gradient-weighted activations. Attention is\n"
            f"  {'concentrated in a small focal area' if analysis['gradcam'].max() > 0.8 else 'spread broadly across the image'}.\n\n"
            f"LIME Analysis:\n"
            f"  Yellow boundary outlines show the superpixels (image segments)\n"
            f"  that most supported the prediction '{label}'.\n"
            f"  The highlighted regions correspond to the hand-gesture area\n"
            f"  that the model considers most discriminative.\n\n"
            f"SHAP — Integrated Gradients Analysis:\n"
            f"  Attribution computed by averaging gradients along 50 interpolation\n"
            f"  steps from a black baseline to the input image. Brighter pixels\n"
            f"  contributed more to the predicted class score.\n"
            f"  Mean attribution: {float(analysis['shap'].mean()):.4f}   "
            f"Max: {float(analysis['shap'].max()):.4f}\n\n"
            f"Occlusion Sensitivity Analysis:\n"
            f"  A 16x16 grey patch was slid over the image (stride 8px).\n"
            f"  Red regions caused the largest confidence drop when covered,\n"
            f"  confirming they are the most decision-critical areas.\n"
            f"  Baseline confidence: {analysis['occ_conf']:.2%}   "
            f"High-sensitivity pixels: {int((analysis['occlusion'] > 0.5).sum())}\n\n"
            f"Alternative Predictions:\n"
            f"{alts}\n"
            f"Decision Basis:\n"
            f"  1. Grad-CAM    — gradient-weighted spatial attention\n"
            f"  2. LIME        — local superpixel importance\n"
            f"  3. SHAP (IG)   — integrated gradient pixel attribution\n"
            f"  4. Occlusion   — perturbation-based importance verification\n"
            f"  5. ResNet18 pattern matching against "
            f"{config.NUM_CLASSES} ISL gesture classes\n\n"
            f"Confidence: >90% very high | 70-90% high | "
            f"50-70% moderate | <50% low"
        )
        ax_txt.text(0.02, 0.97, body, va='top', fontsize=8.5,
                    family='monospace', transform=ax_txt.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue',
                              alpha=0.15))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # ── main report generation ────────────────────────────────────────────────

    def generate_report(self, image_paths, output_pdf=None):
        """
        Generate one PDF for one or more images.
        image_paths: str  OR  list[str]
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        n = len(image_paths)

        if output_pdf is None:
            if n == 1:
                name  = os.path.basename(image_paths[0]).split('.')[0]
                fname = f"xai_report_{name}.pdf"
            else:
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"xai_report_batch_{ts}.pdf"
            output_pdf = os.path.join(config.RESULTS_DIR, "reports", fname)

        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        print(f"\n{'='*70}")
        print(f"GENERATING XAI REPORT  ({n} image{'s' if n > 1 else ''})  "
              f"→  {1 + 2*n} pages")
        print(f"Output: {output_pdf}")
        print(f"{'='*70}")

        with PdfPages(output_pdf) as pdf:
            print("  • Cover page...")
            self._cover_page(pdf, n)

            for i, img_path in enumerate(image_paths, 1):
                print(f"  • Sample {i}/{n}: {os.path.basename(img_path)}")
                img_tensor, original_img = self.load_image(img_path)

                for model_name in self.models:
                    analysis = self.analyze_image(
                        img_tensor, original_img, model_name)
                    self._visuals_page(pdf, img_path, original_img, analysis)
                    self._analysis_page(pdf, img_path, model_name, analysis)

            d = pdf.infodict()
            d['Title']        = 'ISL XAI Report'
            d['Author']       = 'ISL XAI Recognition System'
            d['Subject']      = 'Explainable AI — Indian Sign Language'
            d['Keywords']     = ('XAI, Grad-CAM, LIME, SHAP, '
                                 'Integrated Gradients, Occlusion')
            d['CreationDate'] = datetime.now()

        print(f"\nOK  Report saved: {output_pdf}")
        print(f"    Pages: {1 + 2*n}  (1 cover + 2 per sample)")
        return output_pdf


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="XAI Report — single-image or batch mode")
    parser.add_argument("image_path", nargs='?', default=None,
                        help="Image path (omit for batch mode)")
    parser.add_argument("--output",  help="Output PDF path")
    parser.add_argument("--samples", type=int, default=4,
                        help="Images in batch mode (default 4 -> 9-page PDF)")
    args = parser.parse_args()

    print("="*70)
    print("XAI REPORT GENERATOR")
    print("  Grad-CAM | LIME | SHAP (Integrated Gradients) | Occlusion")
    print("="*70)

    try:
        gen = XAIReportGenerator()
    except Exception as e:
        print(f"\nError loading model: {e}")
        return

    try:
        if args.image_path:
            if not os.path.exists(args.image_path):
                print(f"Image not found: {args.image_path}")
                return
            pdf_path = gen.generate_report(args.image_path, args.output)
            print(f"\nOpen: {pdf_path}")
        else:
            print(f"\nBatch mode — {args.samples} random test images")
            data_obj = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
            _, _, test_paths, _ = data_obj.load_data()

            if not test_paths:
                print("No test data found!")
                return

            indices     = random.sample(range(len(test_paths)),
                                        min(args.samples, len(test_paths)))
            image_paths = [test_paths[i] for i in indices]

            print("Selected images:")
            for p in image_paths:
                print(f"  {p}")

            pdf_path = gen.generate_report(image_paths, args.output)
            print(f"\nOpen: {pdf_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()