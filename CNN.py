import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ===============================
# 1. Arquitectura CNN
# ===============================
class EnemyCNN(nn.Module):
    def __init__(self):
        super(EnemyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Para imágenes de 128x128 → tras 3 poolings (factor 8) quedan 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
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
dataset_path = r"C:\Users\lilia\OneDrive\Escritorio\ahl\Mai\aprendizaje_profundo_b\dataset"  # <--- cambia a tu ruta local

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Clases detectadas:", train_dataset.classes)

# ===============================
# 3. Entrenamiento
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnemyCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ===============================
# 4. Evaluación por carpetas enemy y n_enemy
# ===============================
# Diccionario índice → nombre de clase
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}


print("Train clases:", train_dataset.classes)
print("Test clases:", test_dataset.classes)

model.eval()
correct, total = 0, 0

# Contadores por clase
class_correct = {cls: 0 for cls in test_dataset.classes}
class_total = {cls: 0 for cls in test_dataset.classes}

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()

        # Actualizar conteo global
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Actualizar conteo por clase
        for i in range(len(labels)):
            true_label = int(labels[i].item())
            pred_label = int(predicted[i].item())
            class_total[idx_to_class[true_label]] += 1
            if true_label == pred_label:
                class_correct[idx_to_class[true_label]] += 1

# Precisión global
print(f"\n🔎 Precisión total en test: {100 * correct / total:.2f}%")

# Precisión por clase (enemy vs n_enemy)
for cls in test_dataset.classes:
    if class_total[cls] > 0:
        acc = 100 * class_correct[cls] / class_total[cls]
        print(f"📂 Clase '{cls}': {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

import matplotlib.pyplot as plt
import numpy as np

# Diccionario: índice de clase → nombre de clase (enemigos / no_enemigos)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# Mostrar algunas predicciones
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()

        # Para cada imagen del batch
        for j in range(images.size(0)):
            true_label = int(labels[j].item())
            pred_label = int(predicted[j].item())
            print(f"Imagen {i*test_loader.batch_size + j}: "
                  f"Real = {idx_to_class[true_label]}, "
                  f"Predicho = {idx_to_class[pred_label]}")

            # Mostrar la primera imagen mal clasificada como ejemplo
            if true_label != pred_label:
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = (img * 0.5 + 0.5)  # desnormalizar
                plt.imshow(img)
                plt.title(f"Real: {idx_to_class[true_label]} - Predicho: {idx_to_class[pred_label]}")
                plt.axis("off")
                plt.show()
                break
        break  # quita este break si quieres revisar todo el dataset


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Diccionario de índice → nombre de clase
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

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
