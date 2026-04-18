"""
Training Script — ResNet18 on isl_dataset
Overfitting fixes applied:
  • Label smoothing 0.15 in CrossEntropyLoss
  • AdamW with weight_decay from config (5e-4)
  • CosineAnnealingLR instead of ReduceLROnPlateau
  • Gradient clipping retained
  • Early stopping on val_loss
  • Detailed train/val gap monitoring
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

import config
from data_loader import SimpleDataLoader, create_data_loaders
from model_architecture import create_resnet18_model, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️  Early stopping — no improvement for {self.patience} epochs.")
                print(f"    Best epoch was {self.best_epoch + 1}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'],   'r-', label='Val',   linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'],   'r-', label='Val',   linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Train-Val gap
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 0].plot(epochs, gap, 'g-', linewidth=2)
    axes[1, 0].axhline(0,  color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(10, color='r', linestyle='--', alpha=0.3, label='Overfit threshold')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Gap (%)')
    axes[1, 0].set_title('Train − Val Accuracy Gap')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # LR
    axes[1, 1].plot(epochs, history['learning_rates'], color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f'training_history_{ts}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training history saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    print("=" * 80)
    print("🚀 TRAINING RESNET18 — OVERFITTING-HARDENED")
    print("=" * 80)

    device = config.DEVICE
    print(f"Device : {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # deterministic for reproducibility (already set in config)
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Seed everything ──────────────────────────────────────────────────────
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print(f"\nHyperparameters:")
    print(f"  Batch size    : {config.BATCH_SIZE}")
    print(f"  Epochs        : {config.EPOCHS}")
    print(f"  LR            : {config.LEARNING_RATE}")
    print(f"  Weight decay  : {config.WEIGHT_DECAY}")
    print(f"  Dropout       : {config.DROPOUT_RATE}")
    print(f"  Label smooth  : 0.15")
    print(f"  ES patience   : {config.EARLY_STOPPING_PATIENCE}")
    print("=" * 80)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    print(f"  Train dir : {config.TRAIN_DIR}")
    print(f"  Test dir  : {config.TEST_DIR}")
    loader = SimpleDataLoader(config.TRAIN_DIR, config.TEST_DIR)
    tv_paths, tv_labels, test_paths, test_labels = loader.load_data()

    if not tv_paths:
        print("No training data found!")
        return None

    train_loader, val_loader, test_loader = create_data_loaders(
        tv_paths, tv_labels, test_paths, test_labels
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = create_resnet18_model()

    # Guard against "CUDA device busy / unavailable" at model transfer time.
    # This can happen on WSL / single-GPU machines when another app holds
    # the GPU context (browser with WebGL, a game, another PyTorch process).
    try:
        model.to(device)
    except RuntimeError as e:
        if "busy" in str(e).lower() or "unavailable" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n⚠️  GPU unavailable during model transfer: {e}")
            print("   ─────────────────────────────────────────────────────")
            print("   HOW TO FREE THE GPU:")
            print("   1. Close Chrome / Edge (they use GPU for rendering)")
            print("   2. Close any games or GPU-accelerated apps")
            print("   3. In another terminal run:")
            print("        nvidia-smi   # to see what is using the GPU")
            print("   4. Kill the offending PID:  kill -9 <PID>")
            print("   5. Then re-run:  python train.py")
            print("   ─────────────────────────────────────────────────────")
            print("   Falling back to CPU for this run …\n")
            device = torch.device('cpu')
            model.to(device)
        else:
            raise

    total, trainable = count_parameters(model)
    print(f"\n📊 Parameters — total: {total:,}  trainable: {trainable:,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    # label_smoothing=0.15 softens targets → discourages over-confident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # CosineAnnealingLR decays smoothly to MIN_LR — avoids the plateau jumps
    # that can cause the model to re-memorise after an LR drop
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.MIN_LR,
    )

    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'learning_rates': [],
    }

    best_val_loss = float('inf')
    best_val_acc  = 0.0
    model_path    = os.path.join(config.MODEL_DIR, config.RESNET_MODEL_NAME)

    print(f"\n🎯 Training for up to {config.EPOCHS} epochs …")
    print(f"   Model → {model_path}")
    print("=" * 80)

    training_start = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}   LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}")

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total_samples = 0.0, 0, 0

        pbar = tqdm(train_loader, desc="Train")
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss    += loss.item()
                _, predicted   = outputs.max(1)
                total_samples += labels.size(0)
                correct       += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.3f}',
                    'acc' : f'{100.*correct/total_samples:.1f}%',
                })

                # Periodic memory cleanup
                if batch_idx % 100 == 0 and batch_idx > 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️  OOM at batch {batch_idx} — skipping")
                    torch.cuda.empty_cache()
                    continue
                raise

        train_acc      = 100. * correct / total_samples
        avg_train_loss = train_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(
                    tqdm(val_loader, desc="Val  ")):
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                    val_loss      += loss.item()
                    _, predicted   = outputs.max(1)
                    total_samples += labels.size(0)
                    correct       += predicted.eq(labels).sum().item()

                    if batch_idx % 50 == 0 and batch_idx > 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        continue
                    raise

        val_acc      = 100. * correct / total_samples
        avg_val_loss = val_loss / len(val_loader)

        # Step scheduler once per epoch
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        gap = train_acc - val_acc
        flag = "  ⚠️  OVERFIT WARNING" if gap > 15 else ""

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train  loss={avg_train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"   Val    loss={avg_val_loss:.4f}  acc={val_acc:.2f}%")
        print(f"   Gap    {gap:+.1f}%{flag}")
        print(f"   Time   {time.time()-epoch_start:.1f}s")

        # ── Save best ─────────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc  = val_acc
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':            val_acc,
                'val_loss':           avg_val_loss,
                'train_acc':          train_acc,
                'history':            history,
            }, model_path)
            print(f"   ✅ Best model saved  (val_loss={avg_val_loss:.4f}  val_acc={val_acc:.2f}%)")

        # ── Early stopping ────────────────────────────────────────────────────
        early_stopping(avg_val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\n⛔ Stopped early at epoch {epoch+1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc   = torch.cuda.memory_allocated(0)  / 1e9
            reserved= torch.cuda.memory_reserved(0)   / 1e9
            print(f"   💾 GPU {alloc:.2f}/{reserved:.2f} GB")

    # ── Post-training ─────────────────────────────────────────────────────────
    total_time = time.time() - training_start
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"   Total time        : {total_time/60:.1f} min")
    print(f"   Epochs trained    : {epoch+1}")
    print(f"   Best val accuracy : {best_val_acc:.2f}%")
    print(f"   Best val loss     : {best_val_loss:.4f}")
    print(f"   Model saved to    : {model_path}")

    plot_training_history(history)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join('results', f'training_summary_{ts}.txt')
    os.makedirs('results', exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write("TRAINING SUMMARY\n\n")
        f.write(f"Date              : {datetime.now()}\n")
        f.write(f"Total time        : {total_time/60:.1f} min\n")
        f.write(f"Epochs            : {epoch+1}\n")
        f.write(f"Best epoch        : {early_stopping.best_epoch+1}\n")
        f.write(f"Best val loss     : {best_val_loss:.4f}\n")
        f.write(f"Best val accuracy : {best_val_acc:.2f}%\n")
        f.write(f"Model path        : {model_path}\n")
    print(f"📄 Summary saved: {summary_path}")
    print("\n🎯 Next: run  python test.py")

    return model_path


if __name__ == "__main__":
    try:
        path = train_model()
        if path:
            print(f"\n✅ Model: {path}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        import traceback
        print(f"\n❌ {e}")
        traceback.print_exc()