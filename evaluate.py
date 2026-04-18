"""
Fixed Model Evaluation Script
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import config
from data_loader import SimpleDataLoader, create_data_loaders
from model_architecture import create_resnet18_model


def load_model(model_path, model_class):
    """Load a trained model"""
    model = model_class()
    
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded from checkpoint dictionary")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Model loaded from state dict")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    
    model.to(config.DEVICE)
    model.eval()
    return model


def get_predictions(model, data_loader, device):
    """Get predictions from model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Getting predictions'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_top_k_accuracy(y_probs, y_true, k_values=[1, 3, 5]):
    """Calculate top-k accuracy"""
    top_k_accuracies = {}
    
    for k in k_values:
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        top_k_acc = np.mean(correct) * 100
        top_k_accuracies[f'top_{k}'] = top_k_acc
        print(f"Top-{k} Accuracy: {top_k_acc:.2f}%")
    
    return top_k_accuracies


def plot_top_k_accuracy(top_k_accuracies, model_name):
    """Plot top-k accuracy"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [int(k.split('_')[1]) for k in top_k_accuracies.keys()]
    accuracies = list(top_k_accuracies.values())
    k_labels = [f'Top-{k}' for k in k_values]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax.bar(k_labels, accuracies, color=colors[:len(k_values)], alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model_name} - Top-K Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = f"{config.RESULTS_DIR}/{model_name}_top_k_accuracy.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Top-k accuracy plot saved to: {save_path}")


def evaluate_model_comprehensive(model, test_loader, model_name, device):
    """Comprehensive model evaluation"""
    print("\n" + "="*80)
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print("="*80)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)
    
    # Basic metrics
    print("\n1. Basic Metrics:")
    test_acc = accuracy_score(y_true, y_pred) * 100
    print(f"   - Accuracy: {test_acc:.2f}%")
    
    # Top-k accuracy
    print("\n2. Top-K Accuracy:")
    top_k_accs = calculate_top_k_accuracy(y_probs, y_true, k_values=[1, 3, 5])
    plot_top_k_accuracy(top_k_accs, model_name)
    
    # Classification report
    print("\n3. Classification Report:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.ISL_CLASSES,
        digits=3,
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    print("\n4. Generating Confusion Matrix...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=config.ISL_CLASSES, yticklabels=config.ISL_CLASSES,
                cbar_kws={'label': 'Count'}, annot_kws={"size": 6})
    ax1.set_title(f'{model_name} - Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', ax=ax2,
                xticklabels=config.ISL_CLASSES, yticklabels=config.ISL_CLASSES,
                cbar_kws={'label': 'Proportion'}, annot_kws={"size": 6})
    ax2.set_title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    save_path = f"{config.RESULTS_DIR}/{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {save_path}")
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Test Samples: {len(y_true)}")
    print(f"Overall Accuracy: {test_acc:.2f}%")
    print(f"Top-3 Accuracy: {top_k_accs['top_3']:.2f}%")
    print(f"Top-5 Accuracy: {top_k_accs['top_5']:.2f}%")
    
    results = {
        'model_name': model_name,
        'test_accuracy': test_acc,
        'top_k_accuracies': top_k_accs,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': y_probs
    }
    
    return results


def main():
    """Main evaluation function"""
    
    print("="*80)
    print("MODEL EVALUATION SCRIPT")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    
    # Load data
    print("\nLoading test data...")
    data_loader = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
    train_paths, train_labels, test_paths, test_labels = data_loader.load_data()
    
    if len(test_paths) == 0:
        print("No test data available!")
        return
    
    _, _, test_loader = create_data_loaders(
        train_paths, train_labels, test_paths, test_labels
    )
    
    print(f"Test samples: {len(test_paths)}")
    
    # Evaluate ResNet18 model
    resnet_path = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)
    if os.path.exists(resnet_path):
        print("\nLoading ResNet18 model...")
        try:
            resnet_model = load_model(resnet_path, create_resnet18_model)
            results = evaluate_model_comprehensive(
                resnet_model, test_loader, "ResNet18", config.DEVICE
            )
        except Exception as e:
            print(f"✗ Error evaluating ResNet18: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ ResNet18 model not found at {resnet_path}")
        print("Please run train.py first to train the model.")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()