"""
TRAIN ViT/DeiT MODEL - For ensemble diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = 'd:/Nasir/Vishon Task/output/output'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

CLASSES = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
ORGAN_CLASSES = ['ear', 'nose', 'throat', 'vc']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

with open(os.path.join(TRAIN_DIR, 'data.json')) as f:
    train_data = json.load(f)
with open(os.path.join(TEST_DIR, 'data.json')) as f:
    test_data = json.load(f)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

IMG_SIZE = 224  # ViT default

class HierarchicalModelV1(nn.Module):
    def __init__(self, backbone_name='vit_small_patch16_224', dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.stage1_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True), nn.Dropout(dropout/2),
            nn.Linear(256, len(ORGAN_CLASSES))
        )
        self.ear_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        self.nose_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        self.vc_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, x):
        features = self.backbone(x)
        return self.stage1_head(features), self.ear_head(features), self.nose_head(features), self.vc_head(features)

class MedicalDataset(Dataset):
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
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = class_to_idx[item['anatomical_region']]
        organ = item['anatomical_region'].split('-')[0] if '-' in item['anatomical_region'] else item['anatomical_region']
        organ_idx = ORGAN_CLASSES.index(organ)
        
        sub_label = 0
        if organ in ['ear', 'nose']:
            sub_label = 1 if 'right' in item['anatomical_region'] else 0
        elif organ == 'vc':
            sub_label = 1 if 'open' in item['anatomical_region'] else 0
        
        return image, label, organ_idx, sub_label

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

train_dataset = MedicalDataset(train_data, TRAIN_DIR, train_transform)
test_dataset = MedicalDataset(test_data, TEST_DIR, test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Create model
print("\n" + "="*60)
print("TRAINING ViT SMALL MODEL")
print("="*60)

model = HierarchicalModelV1(backbone_name='vit_small_patch16_224').to(device)
print(f"Feature dim: {model.feature_dim}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
criterion = nn.CrossEntropyLoss()

label_smoothing = 0.1

def train_epoch():
    model.train()
    total_loss = 0
    for images, labels, organs, subs in train_loader:
        images, labels, organs, subs = images.to(device), labels.to(device), organs.to(device), subs.to(device)
        
        optimizer.zero_grad()
        organ_out, ear_out, nose_out, vc_out = model(images)
        
        # Label smoothing
        loss_organ = F.cross_entropy(organ_out, organs, label_smoothing=label_smoothing)
        
        # Subtype losses
        ear_mask = (organs == 0)
        nose_mask = (organs == 1)
        vc_mask = (organs == 3)
        
        loss_ear = F.cross_entropy(ear_out[ear_mask], subs[ear_mask], label_smoothing=label_smoothing) if ear_mask.any() else 0
        loss_nose = F.cross_entropy(nose_out[nose_mask], subs[nose_mask], label_smoothing=label_smoothing) if nose_mask.any() else 0
        loss_vc = F.cross_entropy(vc_out[vc_mask], subs[vc_mask], label_smoothing=label_smoothing) if vc_mask.any() else 0
        
        loss = loss_organ + 0.5 * (loss_ear + loss_nose + loss_vc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, organs, subs in test_loader:
            images, labels = images.to(device), labels.to(device)
            organ_out, ear_out, nose_out, vc_out = model(images)
            
            organ_probs = F.softmax(organ_out, dim=1)
            ear_probs = F.softmax(ear_out, dim=1)
            nose_probs = F.softmax(nose_out, dim=1)
            vc_probs = F.softmax(vc_out, dim=1)
            
            probs = torch.zeros(images.size(0), 7, device=device)
            probs[:, 0] = organ_probs[:, 0] * ear_probs[:, 0]
            probs[:, 1] = organ_probs[:, 0] * ear_probs[:, 1]
            probs[:, 2] = organ_probs[:, 1] * nose_probs[:, 0]
            probs[:, 3] = organ_probs[:, 1] * nose_probs[:, 1]
            probs[:, 4] = organ_probs[:, 2]
            probs[:, 5] = organ_probs[:, 3] * vc_probs[:, 0]
            probs[:, 6] = organ_probs[:, 3] * vc_probs[:, 1]
            
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

best_acc = 0
for epoch in range(1, 61):
    train_loss = train_epoch()
    test_acc = evaluate()
    scheduler.step()
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_acc': test_acc,
            'backbone': 'vit_small_patch16_224',
            'image_size': IMG_SIZE
        }, 'ensemble_vit.pth')
        print(f"Epoch {epoch:2d}: {100*test_acc:.2f}% <- SAVED")
    elif epoch % 10 == 0:
        print(f"Epoch {epoch:2d}: {100*test_acc:.2f}%")

print(f"\nBest: {100*best_acc:.2f}%")
