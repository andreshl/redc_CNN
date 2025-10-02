import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# -----------------------
# 1. Configuración
# -----------------------
data_dir = "dataset"  # carpeta donde están train/val/test
batch_size = 16
num_epochs = 10
learning_rate = 0.001
num_classes = 2  # enemigo / no enemigo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 2. Transformaciones (Data Augmentation)
# -----------------------
train_transforms = transforms.Compose([
    transforms.Resize((640,360)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((640,360)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------
# 3. Carga de datos
# -----------------------
train_dataset = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir,'val'), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# 4. Modelo (ResNet18 Transfer Learning)
# -----------------------
model = models.resnet18(pretrained=True)

# Congelar capas convolucionales para no entrenar desde cero
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la última capa fully connected
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# -----------------------
# 5. Función de pérdida y optimizador
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# -----------------------
# 6. Entrenamiento
# -----------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_dataset)
    train_acc  = correct / total

    # Validación
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_dataset)
    val_acc  = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}")

# -----------------------
# 7. Guardar modelo
# -----------------------
torch.save(model.state_dict(), "resnet18_enemigos.pth")
print("Modelo guardado como resnet18_enemigos.pth")