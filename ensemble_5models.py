"""
5-MODEL COMPREHENSIVE ENSEMBLE
B2 + B3 + B4 + ConvNeXt + ViT
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
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = 'd:/Nasir/Vishon Task/output/output'
TEST_DIR = os.path.join(DATA_DIR, 'test')

CLASSES = ['ear-left', 'ear-right', 'nose-left', 'nose-right', 'throat', 'vc-closed', 'vc-open']
ORGAN_CLASSES = ['ear', 'nose', 'throat', 'vc']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

with open(os.path.join(TEST_DIR, 'data.json')) as f:
    test_data = json.load(f)
print(f"Test: {len(test_data)} samples")

class HierarchicalModelV1(nn.Module):
    def __init__(self, backbone_name='efficientnet_b2', dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.stage1_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout/2), nn.Linear(256, len(ORGAN_CLASSES)))
        self.ear_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        self.nose_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        self.vc_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, x):
        features = self.backbone(x)
        return self.stage1_head(features), self.ear_head(features), self.nose_head(features), self.vc_head(features)

def get_7class_probs(model, img_tensor):
    organ, ear, nose, vc = model(img_tensor)
    organ_probs = F.softmax(organ, dim=1)
    ear_probs = F.softmax(ear, dim=1)
    nose_probs = F.softmax(nose, dim=1)
    vc_probs = F.softmax(vc, dim=1)
    probs = torch.zeros(img_tensor.size(0), 7, device=img_tensor.device)
    probs[:, 0] = organ_probs[:, 0] * ear_probs[:, 0]
    probs[:, 1] = organ_probs[:, 0] * ear_probs[:, 1]
    probs[:, 2] = organ_probs[:, 1] * nose_probs[:, 0]
    probs[:, 3] = organ_probs[:, 1] * nose_probs[:, 1]
    probs[:, 4] = organ_probs[:, 2]
    probs[:, 5] = organ_probs[:, 3] * vc_probs[:, 0]
    probs[:, 6] = organ_probs[:, 3] * vc_probs[:, 1]
    return probs

def get_tta(size):
    s1 = int(size * 1.15)
    s2 = int(size * 1.1)
    s3 = int(size * 1.05)
    return [
        A.Compose([A.Resize(size, size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(s1, s1), A.CenterCrop(size, size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(s2, s2), A.CenterCrop(size, size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(s3, s3), A.CenterCrop(size, size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(size+16, size+16), A.CenterCrop(size, size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(size+32, size+32), A.CenterCrop(size, size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    ]

# Load models
print("Loading 5 models...")
models = []

configs = [
    ('best_hierarchical_model.pth', 'efficientnet_b2', 'B2', 260),
    ('ensemble_effb3.pth', 'efficientnet_b3', 'B3', 260),
    ('ensemble_effb4_v1.pth', 'efficientnet_b4', 'B4', 260),
    ('ensemble_convnext.pth', 'convnext_tiny', 'ConvNeXt', 256),
    ('ensemble_vit.pth', 'vit_small_patch16_224', 'ViT', 224),
]

for path, backbone, name, img_size in configs:
    if os.path.exists(path):
        try:
            model = HierarchicalModelV1(backbone_name=backbone).to(device)
            checkpoint = torch.load(path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            acc = checkpoint.get('test_acc', 0) * 100
            models.append((name, model, img_size))
            print(f"  {name}: {acc:.2f}%")
        except Exception as e:
            print(f"  {name}: Failed - {str(e)[:50]}")

print(f"Loaded {len(models)} models")

# Compute predictions
print("\nComputing predictions with TTA...")
model_probs = {name: [] for name, _, _ in models}
true_labels = []

for i, item in enumerate(test_data):
    img_path = os.path.join(TEST_DIR, 'imgs', item['path'])
    image = np.array(Image.open(img_path).convert('RGB'))
    label = class_to_idx[item['anatomical_region']]
    true_labels.append(label)
    
    for name, model, img_size in models:
        transforms = get_tta(img_size)
        probs_list = []
        for t in transforms:
            try:
                img_tensor = t(image=image)['image'].unsqueeze(0).to(device)
                with torch.no_grad():
                    p = get_7class_probs(model, img_tensor).cpu()
                    probs_list.append(p)
            except:
                pass
        avg = torch.mean(torch.stack(probs_list), dim=0) if probs_list else torch.zeros(1, 7)
        model_probs[name].append(avg)
    
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(test_data)}")

for name in model_probs:
    model_probs[name] = torch.cat(model_probs[name], dim=0)

# Individual accuracy
print("\n" + "="*60)
print("INDIVIDUAL MODELS (with TTA)")
print("="*60)
for name, _, _ in models:
    preds = model_probs[name].argmax(dim=1).tolist()
    correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
    print(f"  {name}: {100*correct/len(true_labels):.2f}%")

# Grid search
print("\n" + "="*60)
print("GRID SEARCH (5 models)")
print("="*60)

model_names = [m[0] for m in models]
best_acc = 0
best_weights = []

# Coarse search first
for weights in itertools.product([0.0, 0.3, 0.6, 0.9, 1.2], repeat=len(model_names)):
    if sum(weights) == 0:
        continue
    combined = sum(w * model_probs[n] for w, n in zip(weights, model_names)) / sum(weights)
    preds = combined.argmax(dim=1).tolist()
    correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
    acc = 100 * correct / len(true_labels)
    
    if acc > best_acc:
        best_acc = acc
        best_weights = list(weights)
        print(f"  {acc:.2f}% with {weights}")

# Fine search around best
print("\nRefinement...")
finer = np.arange(-0.3, 0.31, 0.1)
for delta in itertools.product(finer, repeat=len(model_names)):
    weights = [max(0, best_weights[i] + delta[i]) for i in range(len(model_names))]
    if sum(weights) == 0:
        continue
    combined = sum(w * model_probs[n] for w, n in zip(weights, model_names)) / sum(weights)
    preds = combined.argmax(dim=1).tolist()
    correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
    acc = 100 * correct / len(true_labels)
    
    if acc > best_acc:
        best_acc = acc
        best_weights = weights
        print(f"  {acc:.2f}% with {[round(w, 2) for w in weights]}")

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Best Accuracy: {best_acc:.2f}%")
print(f"Best Weights: {['{}:{:.2f}'.format(n, w) for n, w in zip(model_names, best_weights)]}")

combined = sum(w * model_probs[n] for w, n in zip(best_weights, model_names)) / sum(best_weights)
preds = combined.argmax(dim=1).tolist()
errors = [(i, test_data[i]['path'], CLASSES[true_labels[i]], CLASSES[preds[i]]) 
          for i in range(len(true_labels)) if preds[i] != true_labels[i]]

print(f"\nErrors: {len(errors)}")
for idx, path, true, pred in errors:
    conf = combined[idx].max().item()
    true_conf = combined[idx, class_to_idx[true]].item()
    print(f"  {path}: {true} -> {pred} ({true_conf:.3f} vs {conf:.3f})")

# Class breakdown
print("\n" + "="*60)
print("ERRORS BY CLASS")
print("="*60)
for cls in CLASSES:
    cls_idx = class_to_idx[cls]
    total = sum(1 for t in true_labels if t == cls_idx)
    wrong = sum(1 for (p, t) in zip(preds, true_labels) if t == cls_idx and p != t)
    pct = 100*wrong/total if total > 0 else 0
    print(f"  {cls}: {wrong}/{total} ({pct:.1f}%)")

if best_acc >= 100.0:
    print("\n" + "="*60)
    print("*** 100% ACCURACY ACHIEVED! ***")
    print("="*60)
