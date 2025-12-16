import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ============================================================
# 1. 설정(경로, 하이퍼파라미터, 모델 타입)
# ============================================================

# CSV 경로 (앞에서 만든 split_results 기준)
BASE_DIR   = r"D:\MDO\footpress\result\v2-2048\_lp(x)-rp(y)_v-2048"
TRAIN_CSV  = os.path.join(BASE_DIR, "train.csv")
VAL_CSV    = os.path.join(BASE_DIR, "val.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test.csv")

BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
NUM_CLASSES = 2  # Co vs Pt

# 사용할 모델 종류: "simplecnn" 또는 "resnet50"
MODEL_TYPE = "resnet50"  # 필요하면 "simplecnn"으로 바꿔서도 돌려보기

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# 2. Dataset 클래스 정의
# ============================================================

class FootpressDataset(Dataset):
    """
    CSV 컬럼: path, subject, label
    path  : 이미지 전체 경로
    label : 0 (Co), 1 (Pt)
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label = int(row['label'])

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

# ============================================================
# 3. Transform & DataLoader
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 필요시 정규화 추가 (미리 학습된 ResNet 쓸 때 성능 조금 더 안정)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

train_ds = FootpressDataset(TRAIN_CSV, transform)
val_ds   = FootpressDataset(VAL_CSV, transform)
test_ds  = FootpressDataset(TEST_CSV, transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# ============================================================
# 4. 모델 정의(SimpleCNN / ResNet50)
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 112x112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 56x56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28x28

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14x14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_type: str, num_classes: int):
    model_type = model_type.lower()
    if model_type == "simplecnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)

model = build_model(MODEL_TYPE, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# 5. 학습 / 검증 루프
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)         # [B, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


best_val_acc = 0.0
best_model_path = os.path.join(BASE_DIR, f"best_{MODEL_TYPE}.pt")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    # 베스트 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  👉 New best model saved at {best_model_path} (val_acc={best_val_acc:.3f})")

print("Training finished. Best Val Acc:", best_val_acc)

# ============================================================
# 6. Test 셋 평가
# ============================================================

# 베스트 모델 로드해서 평가하는 게 안전
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded best model from:", best_model_path)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"[Test] Loss: {test_loss:.4f} Acc: {test_acc:.3f}")
