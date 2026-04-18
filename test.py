"""
Testing Script - Comprehensive Model Evaluation
This script loads the trained model and performs complete testing with all metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class performance analysis
- Visualization graphs
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import config
from data_loader import SimpleDataLoader, create_data_loaders
from model_architecture import create_resnet18_model
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
from datetime import datetime


def plot_confusion_matrix(cm, class_names, save_dir='results'):
    """Plot and save confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: {plot_path}")
    plt.close()
    
    return plot_path


def plot_per_class_metrics(precision, recall, f1, class_names, save_dir='results'):
    """Plot per-class precision, recall, and F1 scores"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort by F1 score
    indices = np.argsort(f1)
    sorted_precision = precision[indices]
    sorted_recall = recall[indices]
    sorted_f1 = f1[indices]
    sorted_classes = [class_names[i] for i in indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(class_names) * 0.4))
    
    y_pos = np.arange(len(sorted_classes))
    width = 0.25
    
    # Plot bars
    ax.barh(y_pos - width, sorted_precision, width, label='Precision', alpha=0.8, color='#3498db')
    ax.barh(y_pos, sorted_recall, width, label='Recall', alpha=0.8, color='#2ecc71')
    ax.barh(y_pos + width, sorted_f1, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_classes)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics (Sorted by F1-Score)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1.05])
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f'per_class_metrics_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-class metrics saved: {plot_path}")
    plt.close()
    
    return plot_path


def plot_metric_distribution(precision, recall, f1, save_dir='results'):
    """Plot distribution of metrics across all classes"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Metric Distribution Across Classes', fontsize=16, fontweight='bold')
    
    # Precision distribution
    axes[0].hist(precision, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(precision.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {precision.mean():.3f}')
    axes[0].set_xlabel('Precision', fontsize=12)
    axes[0].set_ylabel('Number of Classes', fontsize=12)
    axes[0].set_title('Precision Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recall distribution
    axes[1].hist(recall, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1].axvline(recall.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {recall.mean():.3f}')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Number of Classes', fontsize=12)
    axes[1].set_title('Recall Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 distribution
    axes[2].hist(f1, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[2].axvline(f1.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {f1.mean():.3f}')
    axes[2].set_xlabel('F1-Score', fontsize=12)
    axes[2].set_ylabel('Number of Classes', fontsize=12)
    axes[2].set_title('F1-Score Distribution', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f'metric_distribution_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Metric distribution saved: {plot_path}")
    plt.close()
    
    return plot_path


def test_model():
    """Comprehensive testing of the trained model"""
    print("="*80)
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("="*80)
    
    device = config.DEVICE
    print(f"Device: {device}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Load model path
    model_path = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("Please run train.py first to train the model")
        return None
    
    print(f"\n📂 Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"\n📊 Model Training Info:")
    print(f"  Trained for: {checkpoint['epoch'] + 1} epochs")
    print(f"  Training Accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Create and load model
    print(f"\n🔨 Creating model architecture...")
    model = create_resnet18_model()
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load test data — uses isl_dataset/Test/ folder directly
    print(f"\nLoading test data from: {config.TEST_DIR}")
    data_loader = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
    train_paths, train_labels, test_paths, test_labels = data_loader.load_data()
    
    if len(test_paths) == 0:
        print("❌ No test data found!")
        return None
    
    _, _, test_loader = create_data_loaders(
        train_paths, train_labels, test_paths, test_labels
    )
    
    print(f"✅ Test data loaded: {len(test_paths)} samples in {len(test_loader)} batches")
    
    # ========== TESTING ==========
    print(f"\n{'='*80}")
    print("🔬 RUNNING COMPREHENSIVE TESTING")
    print(f"{'='*80}")
    
    all_preds = []
    all_labels = []
    all_probs = []
    class_correct = {}
    class_total = {}
    
    test_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Get predictions and probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Track per-class accuracy
                for pred, label in zip(predicted, labels):
                    label_idx = label.item()
                    if label_idx not in class_total:
                        class_total[label_idx] = 0
                        class_correct[label_idx] = 0
                    class_total[label_idx] += 1
                    if pred == label:
                        class_correct[label_idx] += 1
                
                # Memory management
                if batch_idx % 20 == 0 and batch_idx > 0:
                    del images, labels, outputs, probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e) or "illegal memory access" in str(e):
                    print(f"\n⚠️ CUDA Error at batch {batch_idx}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    continue
                else:
                    raise e
    
    test_time = time.time() - test_start
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ========== CALCULATE METRICS ==========
    print(f"\n{'='*80}")
    print("📊 CALCULATING PERFORMANCE METRICS")
    print(f"{'='*80}")
    
    # Overall accuracy
    test_accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    # Macro metrics (unweighted average)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Weighted metrics (class-balanced)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # ========== PRINT RESULTS ==========
    print(f"\n{'='*80}")
    print("🎯 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Test Accuracy:           {test_accuracy:.2f}%")
    print(f"Test Time:               {test_time:.2f}s")
    print(f"Samples Tested:          {len(all_labels)}")
    print(f"Samples per Second:      {len(all_labels)/test_time:.1f}")
    
    print(f"\n{'='*80}")
    print("📈 MACRO-AVERAGED METRICS (Equal weight per class)")
    print(f"{'='*80}")
    print(f"Precision (Macro):       {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"Recall (Macro):          {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"F1-Score (Macro):        {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print("⚖️  WEIGHTED-AVERAGED METRICS (Weighted by support)")
    print(f"{'='*80}")
    print(f"Precision (Weighted):    {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    print(f"Recall (Weighted):       {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    print(f"F1-Score (Weighted):     {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    
    # Generalization analysis
    print(f"\n{'='*80}")
    print("🔍 GENERALIZATION ANALYSIS")
    print(f"{'='*80}")
    train_val_gap = checkpoint['train_acc'] - checkpoint['val_acc']
    val_test_gap = checkpoint['val_acc'] - test_accuracy
    print(f"Train-Val Gap:           {train_val_gap:+.2f}%")
    print(f"Val-Test Gap:            {val_test_gap:+.2f}%")
    
    if abs(train_val_gap) < 5 and abs(val_test_gap) < 5:
        print("✅ Excellent generalization! Model performs consistently across all splits.")
    elif abs(train_val_gap) < 10 and abs(val_test_gap) < 10:
        print("✅ Good generalization. Small performance differences are acceptable.")
    else:
        if train_val_gap > 10:
            print("⚠️  Warning: Large train-validation gap suggests overfitting.")
        if abs(val_test_gap) > 10:
            print("⚠️  Warning: Large val-test gap. Validation set may not be representative.")
    
    # ========== PER-CLASS ANALYSIS ==========
    print(f"\n{'='*80}")
    print("📋 PER-CLASS PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Create per-class accuracy dict
    per_class_acc = {idx: (class_correct[idx] / class_total[idx] * 100)
                     for idx in class_total.keys()}
    
    # Sort by F1-score
    class_f1_pairs = [(idx, f1_per_class[idx]) for idx in range(len(config.ISL_CLASSES))]
    sorted_by_f1 = sorted(class_f1_pairs, key=lambda x: x[1])
    
    print("\n📉 WORST 10 PERFORMING CLASSES:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10} {'Support':>10}")
    print("-" * 82)
    for idx, _ in sorted_by_f1[:10]:
        class_name = config.ISL_CLASSES[idx]
        prec = precision_per_class[idx]
        rec = recall_per_class[idx]
        f1 = f1_per_class[idx]
        acc = per_class_acc.get(idx, 0)
        support = class_total.get(idx, 0)
        print(f"{class_name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {acc:>9.2f}% {support:>10}")
    
    print("\n📈 BEST 10 PERFORMING CLASSES:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10} {'Support':>10}")
    print("-" * 82)
    for idx, _ in sorted_by_f1[-10:][::-1]:
        class_name = config.ISL_CLASSES[idx]
        prec = precision_per_class[idx]
        rec = recall_per_class[idx]
        f1 = f1_per_class[idx]
        acc = per_class_acc.get(idx, 0)
        support = class_total.get(idx, 0)
        print(f"{class_name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {acc:>9.2f}% {support:>10}")
    
    # ========== GENERATE VISUALIZATIONS ==========
    print(f"\n{'='*80}")
    print("📊 GENERATING VISUALIZATION PLOTS")
    print(f"{'='*80}")
    
    plot_confusion_matrix(cm, config.ISL_CLASSES)
    plot_per_class_metrics(precision_per_class, recall_per_class, f1_per_class, config.ISL_CLASSES)
    plot_metric_distribution(precision_per_class, recall_per_class, f1_per_class)
    
    # ========== SAVE DETAILED REPORT ==========
    print(f"\n{'='*80}")
    print("💾 SAVING DETAILED TEST REPORT")
    print(f"{'='*80}")
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join('results', f'test_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TEST REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Best Training Epoch: {checkpoint['epoch'] + 1}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Test Accuracy:           {test_accuracy:.2f}%\n")
        f.write(f"Test Time:               {test_time:.2f}s\n")
        f.write(f"Precision (Macro):       {precision_macro:.4f}\n")
        f.write(f"Recall (Macro):          {recall_macro:.4f}\n")
        f.write(f"F1-Score (Macro):        {f1_macro:.4f}\n")
        f.write(f"Precision (Weighted):    {precision_weighted:.4f}\n")
        f.write(f"Recall (Weighted):       {recall_weighted:.4f}\n")
        f.write(f"F1-Score (Weighted):     {f1_weighted:.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("GENERALIZATION\n")
        f.write("="*80 + "\n")
        f.write(f"Training Accuracy:       {checkpoint['train_acc']:.2f}%\n")
        f.write(f"Validation Accuracy:     {checkpoint['val_acc']:.2f}%\n")
        f.write(f"Test Accuracy:           {test_accuracy:.2f}%\n")
        f.write(f"Train-Val Gap:           {train_val_gap:+.2f}%\n")
        f.write(f"Val-Test Gap:            {val_test_gap:+.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(classification_report(all_labels, all_preds,
                                     target_names=config.ISL_CLASSES,
                                     digits=4,
                                     zero_division=0))
        
        f.write("\n" + "="*80 + "\n")
        f.write("PER-CLASS DETAILED METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10} {'Support':>10}\n")
        f.write("-" * 82 + "\n")
        for idx in range(len(config.ISL_CLASSES)):
            class_name = config.ISL_CLASSES[idx]
            prec = precision_per_class[idx]
            rec = recall_per_class[idx]
            f1 = f1_per_class[idx]
            acc = per_class_acc.get(idx, 0)
            support = class_total.get(idx, 0)
            f.write(f"{class_name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {acc:>9.2f}% {support:>10}\n")
    
    print(f"✅ Test report saved: {report_path}")
    
    # Save results as numpy arrays for further analysis
    results_data_path = os.path.join('results', f'test_results_{timestamp}.npz')
    np.savez(results_data_path,
             predictions=all_preds,
             labels=all_labels,
             probabilities=all_probs,
             confusion_matrix=cm)
    print(f"✅ Test results data saved: {results_data_path}")
    
    # ========== FINAL SUMMARY ==========
    print(f"\n{'='*80}")
    print("✅ TESTING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Key Results:")
    print(f"  Test Accuracy:     {test_accuracy:.2f}%")
    print(f"  Macro F1-Score:    {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Weighted F1-Score: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"\n📁 All results saved in 'results/' directory:")
    print(f"  - Test report (TXT)")
    print(f"  - Confusion matrix (PNG)")
    print(f"  - Per-class metrics (PNG)")
    print(f"  - Metric distributions (PNG)")
    print(f"  - Raw results data (NPZ)")
    print(f"{'='*80}\n")
    
    results = {
        'test_accuracy': test_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
        'test_time': test_time,
    }
    
    return results


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🧪 STARTING MODEL TESTING")
        print("="*80 + "\n")
        
        results = test_model()
        
        if results:
            print("\n✅ Testing completed successfully!")
            print(f"Final Test Accuracy: {results['test_accuracy']:.2f}%")
        else:
            print("\n❌ Testing failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()