"""
Train ConvNeXt-Tiny for ensemble
Uses hierarchical architecture with label-aware flip
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
DATA_DIR = 'd:/Nasir/Vishon Task/output/output'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Load data
def load_data(data_dir):
    with open(os.path.join(data_dir, 'data.json')) as f:
        data = json.load(f)
    return data

train_data = load_data(TRAIN_DIR)
test_data = load_data(TEST_DIR)
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Classes
CLASSES = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

# Hierarchical labels
REGION_CLASSES = ['ear', 'nose', 'throat', 'vc']
SIDE_CLASSES = ['left', 'right', 'none']

region_to_idx = {c: i for i, c in enumerate(REGION_CLASSES)}
side_to_idx = {c: i for i, c in enumerate(SIDE_CLASSES)}

def parse_class(class_name):
    if class_name == 'ear-left':
        return 'ear', 'left'
    elif class_name == 'ear-right':
        return 'ear', 'right'
    elif class_name == 'nose-left':
        return 'nose', 'left'
    elif class_name == 'nose-right':
        return 'nose', 'right'
    elif class_name == 'throat':
        return 'throat', 'none'
    elif class_name == 'vc-closed':
        return 'vc', 'none'
    elif class_name == 'vc-open':
        return 'vc', 'none'

# Augmentation
base_transform = A.Compose([
    A.RandomResizedCrop(size=(288, 384), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(288, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Dataset with label-aware flip
class HierarchicalDataset(Dataset):
    def __init__(self, data, img_dir, transform=None, train=True):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, 'imgs', item['path'])
        image = np.array(Image.open(img_path).convert('RGB'))
        
        full_class = item['anatomical_region']
        region, side = parse_class(full_class)
        
        # Label-aware horizontal flip
        apply_flip = self.train and random.random() < 0.5
        if apply_flip:
            image = np.fliplr(image).copy()
            if side == 'left':
                side = 'right'
            elif side == 'right':
                side = 'left'
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        region_label = region_to_idx[region]
        side_label = side_to_idx[side]
        full_label = class_to_idx[full_class]
        
        return image, region_label, side_label, full_label

# Create datasets
train_dataset = HierarchicalDataset(train_data, TRAIN_DIR, base_transform, train=True)
test_dataset = HierarchicalDataset(test_data, TEST_DIR, val_transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)  # Smaller batch for larger model
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

# Hierarchical Model
class HierarchicalModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', num_regions=4, num_sides=3, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        
        self.region_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_regions)
        )
        
        self.side_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_sides)
        )
        
        self.full_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        region_out = self.region_head(features)
        side_out = self.side_head(features)
        full_out = self.full_head(features)
        return region_out, side_out, full_out

print("\n" + "="*60)
print("TRAINING ConvNeXt-Tiny")
print("="*60)

model = HierarchicalModel(backbone_name='convnext_tiny').to(device)

criterion_region = nn.CrossEntropyLoss()
criterion_side = nn.CrossEntropyLoss()
criterion_full = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_acc = 0
for epoch in range(50):
    # Train
    model.train()
    train_correct = 0
    train_total = 0
    
    for images, region_labels, side_labels, full_labels in train_loader:
        images = images.to(device)
        region_labels = region_labels.to(device)
        side_labels = side_labels.to(device)
        full_labels = full_labels.to(device)
        
        optimizer.zero_grad()
        region_out, side_out, full_out = model(images)
        
        loss_region = criterion_region(region_out, region_labels)
        loss_side = criterion_side(side_out, side_labels)
        loss_full = criterion_full(full_out, full_labels)
        
        loss = 0.2 * loss_region + 0.2 * loss_side + 0.6 * loss_full
        
        loss.backward()
        optimizer.step()
        
        _, predicted = full_out.max(1)
        train_total += full_labels.size(0)
        train_correct += predicted.eq(full_labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    scheduler.step()
    
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, _, _, full_labels in test_loader:
            images, full_labels = images.to(device), full_labels.to(device)
            _, _, full_out = model(images)
            _, predicted = full_out.max(1)
            test_total += full_labels.size(0)
            test_correct += predicted.eq(full_labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    if (epoch + 1) % 10 == 0 or test_acc > best_acc:
        print(f"Epoch {epoch+1:2d}: Train: {train_acc:.2f}%, Test: {test_acc:.2f}%", flush=True)
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'ensemble_convnext.pth')
        print(f"  -> New best! Saved to ensemble_convnext.pth", flush=True)

print(f"\nBest ConvNeXt-Tiny accuracy: {best_acc:.2f}%")
print("Model saved as: ensemble_convnext.pth")
