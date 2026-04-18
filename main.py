#!/usr/bin/env python3
"""
Main script for XAI ISL Recognition
Options: evaluate, SHAP, Grad-CAM, LIME, Occlusion, predict, PDF report, run-all pipeline
"""

import os
import sys


def check_environment():
    print("Checking environment …")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ GPU : {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        else:
            print("⚠ No GPU — using CPU")
        return True
    except ImportError:
        print("✗ PyTorch not installed!")
        return False


def find_example_images(search_dir='isl_dataset', n=5):
    """Return up to n example image paths from the dataset."""
    examples = []
    if not os.path.exists(search_dir):
        return examples
    for class_dir in sorted(os.listdir(search_dir))[:n]:
        class_path = os.path.join(search_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        imgs = [f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if imgs:
            examples.append(os.path.join(class_path, imgs[0]))
    return examples


def ask_image_path():
    examples = find_example_images()
    if examples:
        print("\nExample images:")
        for i, p in enumerate(examples, 1):
            print(f"  {i}. {p}")
    path = input("\nEnter image path (or press Enter for first example): ").strip()
    if not path and examples:
        path = examples[0]
    return path


def run_all_xai():
    """Run evaluate → Grad-CAM → LIME → SHAP → Occlusion → PDF report."""
    print("\n" + "="*80)
    print("STEP 1/6 : EVALUATION")
    print("="*80)
    os.system("python evaluate.py")

    print("\n" + "="*80)
    print("STEP 2/6 : GRAD-CAM EXPLANATIONS")
    print("="*80)
    os.system("python explain_gradcam.py")

    print("\n" + "="*80)
    print("STEP 3/6 : LIME EXPLANATIONS")
    print("="*80)
    os.system("python explain_lime.py")

    print("\n" + "="*80)
    print("STEP 4/6 : SHAP / GRADIENT IMPORTANCE (batch)")
    print("="*80)
    os.system("python explain_shap.py")

    print("\n" + "="*80)
    print("STEP 5/6 : OCCLUSION SENSITIVITY (batch)")
    print("="*80)
    os.system("python explain_occlusion.py")

    print("\n" + "="*80)
    print("STEP 6/6 : PDF REPORT (batch — 3 random images)")
    print("="*80)
    os.system("python xai_report.py")

    print("\n" + "="*80)
    print("OK FULL XAI PIPELINE COMPLETE")
    print("="*80)
    print("\nResults saved in:")
    print("   results/                          — evaluation metrics & plots")
    print("   results/xai_explanations/gradcam/ — Grad-CAM")
    print("   results/xai_explanations/lime/    — LIME")
    print("   results/xai_explanations/shap/    — SHAP")
    print("   results/xai_explanations/occlusion/ — Occlusion Sensitivity")
    print("   results/reports/                  — PDF reports")


def main():
    print("=" * 80)
    print("XAI-BASED INDIAN SIGN LANGUAGE RECOGNITION")
    print("=" * 80)

    if not check_environment():
        return

    print("\nAvailable operations:")
    print("\n=== EVALUATION ===")
    print("  1. Evaluate model (metrics + confusion matrix)")
    print("\n=== XAI METHODS ===")
    print("  2. SHAP / Gradient importance  (batch or single image)")
    print("  3. Grad-CAM explanations       (batch or single image)")
    print("  4. LIME explanations           (batch or single image)")
    print("  5. Occlusion Sensitivity       (batch or single image)")
    print("\n=== PREDICTION ===")
    print("  6. Predict single image  (all XAI methods)")
    print("  7. Generate PDF report   (all XAI methods)")
    print("\n=== PIPELINE ===")
    print("  8. Run ALL  (evaluate → Grad-CAM → LIME → SHAP → Occlusion → PDF report)")
    print("=" * 80)

    try:
        choice = input("\nEnter choice (1-8): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received. Exiting.")
        return

    # ── Option 1 : Evaluate ──────────────────────────────────────────────────
    if choice == "1":
        print("\n" + "="*80)
        print("EVALUATING MODEL")
        print("="*80)
        os.system("python evaluate.py")

    # ── Option 2 : SHAP ──────────────────────────────────────────────────────
    elif choice == "2":
        print("\n" + "="*80)
        print("SHAP / GRADIENT IMPORTANCE")
        print("="*80)
        mode = input("Mode — (1) batch test set  (2) single image  [1]: ").strip()
        if mode == "2":
            image_path = ask_image_path()
            if image_path:
                os.system(f'python explain_shap.py "{image_path}"')
        else:
            os.system("python explain_shap.py")

    # ── Option 3 : Grad-CAM ──────────────────────────────────────────────────
    elif choice == "3":
        print("\n" + "="*80)
        print("GRAD-CAM EXPLANATIONS")
        print("="*80)
        mode = input("Mode — (1) batch test set  (2) single image  [1]: ").strip()
        if mode == "2":
            image_path = ask_image_path()
            if image_path:
                os.system(f'python explain_gradcam.py "{image_path}"')
        else:
            os.system("python explain_gradcam.py")

    # ── Option 4 : LIME ──────────────────────────────────────────────────────
    elif choice == "4":
        print("\n" + "="*80)
        print("LIME EXPLANATIONS")
        print("="*80)
        mode = input("Mode — (1) batch test set  (2) single image  [1]: ").strip()
        if mode == "2":
            image_path = ask_image_path()
            if image_path:
                os.system(f'python explain_lime.py "{image_path}"')
        else:
            os.system("python explain_lime.py")

    # ── Option 5 : Occlusion Sensitivity ─────────────────────────────────────
    elif choice == "5":
        print("\n" + "="*80)
        print("OCCLUSION SENSITIVITY")
        print("="*80)
        mode = input("Mode — (1) batch test set  (2) single image  [1]: ").strip()
        if mode == "2":
            image_path = ask_image_path()
            if image_path:
                os.system(f'python explain_occlusion.py "{image_path}"')
        else:
            os.system("python explain_occlusion.py")

    # ── Option 6 : Predict + all XAI ─────────────────────────────────────────
    elif choice == "6":
        print("\n" + "="*80)
        print("PREDICT WITH FULL XAI")
        print("="*80)
        image_path = ask_image_path()
        if image_path:
            os.system(f'python predict.py "{image_path}"')

    # ── Option 7 : PDF report ────────────────────────────────────────────────
    elif choice == "7":
        print("\n" + "="*80)
        print("GENERATE PDF REPORT")
        print("="*80)
        mode = input("Mode — (1) batch test set  (2) single image  [1]: ").strip()
        if mode == "2":
            image_path = ask_image_path()
            if image_path:
                os.system(f'python xai_report.py "{image_path}"')
        else:
            os.system("python xai_report.py")

    # ── Option 8 : Run ALL ───────────────────────────────────────────────────
    elif choice == "8":
        run_all_xai()

    else:
        print("Invalid choice.")
        return

    print("\n" + "="*80)
    print("OPERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()