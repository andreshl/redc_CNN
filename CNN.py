import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# 1. DATASET PERSONALIZADO
# ============================================================================
class DarkColonyDataset(Dataset):
    """
    Dataset para imágenes de Dark Colony
    Estructura esperada:
        data/
            con_enemigos/
                imagen1.jpg
                imagen2.jpg
            sin_enemigos/
                imagen3.jpg
                imagen4.jpg
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# 2. ARQUITECTURAS CNN
# ============================================================================

# 2.2 Red Simple con Operaciones Básicas (Convolución, Pooling, Padding, Stride)
class SimpleCNN(nn.Module):
    """
    Arquitectura simple demostrando operaciones básicas de CNN:
    - Convolución con diferentes kernel sizes
    - Max Pooling para reducción dimensional
    - Padding para mantener dimensiones
    - Stride para control de reducción
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Capa 1: Convolución con padding='same' (mantiene dimensiones)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce a la mitad
        
        # Capa 2: Convolución con stride=2 (reduce dimensiones)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capa 3: Convolución más profunda
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas fully connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 40, 256)  # Ajustar según tamaño de entrada
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu1(self.fc1(x)))
        x = self.fc2(x)
        return x


# 2.3 Arquitectura inspirada en VGG (Bloques repetitivos)
class VGGLikeCNN(nn.Module):
    """
    Arquitectura inspirada en VGG con bloques convolucionales repetitivos
    Características de VGG:
    - Múltiples capas conv de 3x3
    - Bloques con mismo número de filtros
    - MaxPooling después de cada bloque
    """
    def __init__(self, num_classes=2):
        super(VGGLikeCNN, self).__init__()
        
        # Bloque 1 (similar a VGG)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 40, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# 2.3 Arquitectura con Bloques Residuales (inspirada en ResNet)
class ResidualBlock(nn.Module):
    """Bloque residual básico de ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)  # Skip connection
        out = self.relu(out)
        
        return out


class ResNetLikeCNN(nn.Module):
    """
    Arquitectura inspirada en ResNet con conexiones residuales
    Ventaja: Permite entrenar redes más profundas evitando vanishing gradient
    """
    def __init__(self, num_classes=2):
        super(ResNetLikeCNN, self).__init__()
        
        # Capa inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=20, device='cuda'):
    """
    Entrena el modelo y guarda métricas
    """
    model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Fase de validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
    
    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evalúa el modelo en el conjunto de prueba
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy, all_preds, all_labels


def predict_single_image(model, image_path, transform, device='cuda'):
    """
    Predice si hay enemigos en una imagen individual
    """
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = output.max(1)
    
    class_names = ['Sin Enemigos', 'Con Enemigos']
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item() * 100
    
    return predicted_class, confidence


# ============================================================================
# 4. SCRIPT PRINCIPAL
# ============================================================================

def main():
    # Configuración
    IMG_SIZE = (640, 360)  # Ajustar según tus imágenes
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Usando dispositivo: {DEVICE}')
    
    # Transformaciones y Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar datos (ajusta estas rutas a tu estructura)
    # Ejemplo de carga manual de rutas
    image_paths = []
    labels = []
    
    # Cargar imágenes CON enemigos (label=1)
    enemigos_dir = 'dataset/enemys/'
    for img_name in os.listdir(enemigos_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(enemigos_dir, img_name))
            labels.append(1)
    
    # Cargar imágenes SIN enemigos (label=0)
    sin_enemigos_dir = 'dataset/n_enemys/'
    for img_name in os.listdir(sin_enemigos_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(sin_enemigos_dir, img_name))
            labels.append(0)
    
    # Split de datos
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Crear datasets
    train_dataset = DarkColonyDataset(train_paths, train_labels, train_transform)
    val_dataset = DarkColonyDataset(val_paths, val_labels, test_transform)
    test_dataset = DarkColonyDataset(test_paths, test_labels, test_transform)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Seleccionar arquitectura
    print("\nSelecciona la arquitectura:")
    print("1. SimpleCNN (Operaciones básicas)")
    print("2. VGGLikeCNN (Bloques repetitivos)")
    print("3. ResNetLikeCNN (Conexiones residuales)")
    
    # Para este ejemplo, usamos ResNetLikeCNN
    model = ResNetLikeCNN(num_classes=2)
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler para ajustar learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\nModelo seleccionado: {model.__class__.__name__}")
    print(f"Número de parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenar modelo (descomentar cuando tengas los datos)
    """
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=NUM_EPOCHS, device=DEVICE
    )
    
    # Evaluar en test
    test_acc, preds, labels = evaluate_model(model, test_loader, device=DEVICE)
    
    # Guardar modelo
    torch.save(model.state_dict(), 'dark_colony_detector.pth')
    
    # Ejemplo de predicción individual
    predicted_class, confidence = predict_single_image(
        model, 'path/to/test/image.jpg', test_transform, device=DEVICE
    )
    print(f'Predicción: {predicted_class} (Confianza: {confidence:.2f}%)')
    """
    
    print("\n¡Código listo! Ajusta las rutas de datos y descomenta las secciones de entrenamiento.")


if __name__ == '__main__':
    main()