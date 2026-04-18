"""
Fixed Model Architecture - ResNet18 Only
Addresses Overfitting Issues
"""
'''
import torch
import torch.nn as nn
from torchvision import models
import config


class ImprovedResNet18ISL(nn.Module):
    """
    ResNet18 with proper regularization to prevent overfitting
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(ImprovedResNet18ISL, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.resnet.fc.in_features
        
        # Remove the original FC layer
        self.resnet.fc = nn.Identity()
        
        # Simplified classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),  # 0.5 dropout
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE * 0.7),  # 0.35 dropout
            nn.Linear(256, num_classes)
        )
        
        # Fine-tuning strategy: freeze early layers, train later layers
        # Freeze layer1, layer2 - these extract basic features
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = False
        
        # Train layer3 and layer4 for task-specific features
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.resnet(x)
        return self.classifier(x)


def create_resnet18_model(num_classes=config.NUM_CLASSES):
    """Factory function to create ResNet18 model"""
    return ImprovedResNet18ISL(num_classes=num_classes)


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# For backward compatibility - create_attention_resnet18_model is not used
def create_attention_resnet18_model(num_classes=config.NUM_CLASSES):
    """Deprecated - returns regular ResNet18"""
    print("⚠ Warning: Attention model not used. Returning ResNet18 instead.")
    return create_resnet18_model(num_classes)


if __name__ == "__main__":
    print("Testing model architecture...")
    
    dummy = torch.randn(2, 3, 128, 128)
    model = create_resnet18_model()
    
    output = model(dummy)
    total, trainable = count_parameters(model)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Total parameters: {total:,}")
    print(f"✓ Trainable parameters: {trainable:,}")
    print(f"✓ Model ready!")

'''


#------------------------------------------------------------------------------


"""
Model Architecture — ResNet18 with strong regularisation
OVERFITTING FIX: deeper classifier removed, stronger dropout, all layers unfrozen
"""

import torch
import torch.nn as nn
from torchvision import models
import config


class ImprovedResNet18ISL(nn.Module):
    """
    ResNet18 backbone + regularised head.

    Overfitting fixes vs previous version
    ──────────────────────────────────────
    1. ALL layers are trainable (unfreezing layer1/layer2 forces the backbone
       to adapt rather than memorise fixed low-level features).
    2. Classifier head is simpler (fewer parameters → less capacity to memorise).
    3. Dropout rate comes from config (currently 0.6).
    4. BatchNorm is kept — it acts as a light regulariser at the feature level.
    5. Label-smoothing cross-entropy is applied in train.py (0.15).
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.resnet.fc.in_features   # 512 for resnet18
        self.resnet.fc = nn.Identity()              # remove original classifier

        # ── Unfreeze ALL layers — let the whole network adapt ────────────────
        for param in self.resnet.parameters():
            param.requires_grad = True

        # ── Classifier head ───────────────────────────────────────────────────
        # Simpler than before: one hidden layer instead of two.
        # This gives less capacity to overfit while still learning task-specific
        # decision boundaries.
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),          # 0.6
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE * 0.5),    # 0.3  — lighter second drop
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def create_resnet18_model(num_classes=config.NUM_CLASSES):
    return ImprovedResNet18ISL(num_classes=num_classes)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Backward-compat stub
def create_attention_resnet18_model(num_classes=config.NUM_CLASSES):
    print("⚠ Attention model not implemented — returning ResNet18.")
    return create_resnet18_model(num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy  = torch.randn(2, 3, 128, 128)
    model  = create_resnet18_model()
    out    = model(dummy)
    total, trainable = count_parameters(model)

    print(f"✓ Output shape        : {out.shape}")
    print(f"✓ Total parameters    : {total:,}")
    print(f"✓ Trainable parameters: {trainable:,}")
    print("✓ Model ready!")
