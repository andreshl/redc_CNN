# ===============================
# Imports
# ===============================
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 1. Arquitectura CNN
# ===============================
class EnemyCNN(nn.Module):
    def __init__(self):
        super(EnemyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128*16*16, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# ===============================
# 2. Configuración dataset
# ===============================
dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=test_transform)

# ===============================
# 2a. WeightedRandomSampler para balancear clases
# ===============================
targets = [label for _, label in train_dataset]
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
samples_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train clases:", train_dataset.classes)
print("Test clases:", test_dataset.classes)

# ===============================
# 3. Entrenamiento
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnemyCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 22
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    class_correct = {cls: 0 for cls in train_dataset.classes}
    class_total = {cls: 0 for cls in train_dataset.classes}
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Conteo por clase
        predicted = (outputs > 0.5).float()
        for i in range(len(labels)):
            true_label = int(labels[i].item())
            pred_label = int(predicted[i].item())
            class_total[train_dataset.classes[true_label]] += 1
            if true_label == pred_label:
                class_correct[train_dataset.classes[true_label]] += 1
                
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    for cls in train_dataset.classes:
        if class_total[cls] > 0:
            acc = 100 * class_correct[cls] / class_total[cls]
            print(f"  📌 Train clase '{cls}': {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

# ===============================
# 4. Evaluación en test
# ===============================
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
model.eval()
correct, total = 0, 0
class_correct = {cls: 0 for cls in test_dataset.classes}
class_total = {cls: 0 for cls in test_dataset.classes}

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)):
            true_label = int(labels[i].item())
            pred_label = int(predicted[i].item())
            class_total[idx_to_class[true_label]] += 1
            if true_label == pred_label:
                class_correct[idx_to_class[true_label]] += 1
                
print(f"\n🔎 Precisión total en test: {100*correct/total:.2f}%")
for cls in test_dataset.classes:
    if class_total[cls] > 0:
        acc = 100 * class_correct[cls] / class_total[cls]
        print(f"📂 Clase '{cls}': {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

# ===============================
# 5. Predicciones por imagen
# ===============================
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()

        for j in range(images.size(0)):
            true_label = int(labels[j].item())
            pred_label = int(predicted[j].item())
            print(f"Imagen {i*test_loader.batch_size + j}: "
                  f"Real = {idx_to_class[true_label]}, "
                  f"Predicho = {idx_to_class[pred_label]}")

# ===============================
# 6. Reporte y Matriz de confusión
# ===============================
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convertir a enteros
all_preds = np.array(all_preds).astype(int).flatten()
all_labels = np.array(all_labels).astype(int).flatten()

print("\n📊 Reporte de clasificación por clase:")
print(classification_report(all_labels, all_preds, target_names=list(idx_to_class.values())))

print("📌 Matriz de confusión:")
print(confusion_matrix(all_labels, all_preds))
