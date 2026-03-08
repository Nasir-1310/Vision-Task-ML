"""
STEP 1: Baseline Experiment
ResNet-50 with flat 7-class classification
Expected accuracy: ~83%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from collections import Counter

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
print(f"Classes: {CLASSES}")

# Check class distribution
train_labels = [item['anatomical_region'] for item in train_data]
print(f"Train distribution: {Counter(train_labels)}")

# Transforms - NO horizontal flip (preserves left/right)
train_transform = A.Compose([
    A.Resize(480, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(480, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Dataset
class MedicalImageDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, 'imgs', item['path'])
        image = np.array(Image.open(img_path).convert('RGB'))
        label = class_to_idx[item['anatomical_region']]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

# Create datasets and loaders
train_dataset = MedicalImageDataset(train_data, TRAIN_DIR, train_transform)
test_dataset = MedicalImageDataset(test_data, TEST_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Model - ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
print("\n" + "="*60)
print("TRAINING BASELINE MODEL (ResNet-50)")
print("="*60)

best_acc = 0
for epoch in range(30):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    print(f"Epoch {epoch+1:2d}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%", flush=True)
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  -> New best! Saved model", flush=True)

print(f"\nBest Baseline Accuracy: {best_acc:.2f}%")
print("Model saved as: best_model.pth")
