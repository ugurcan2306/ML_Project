import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# --- 1. SETTINGS ---
# TIP: If you move this folder out of OneDrive to C:\plant_data, it will be 10x faster!
BATCH_SIZE = 64  # Increased for your RTX 3060
EPOCHS = 10

class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))    
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Using your specific folders for Train and Valid
    train_dir = r'C:\Users\prohe\OneDrive\Desktop\PlantDiseaseDataset\train'
    valid_dir = r'C:\Users\prohe\OneDrive\Desktop\PlantDiseaseDataset\valid'

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = PlantCNN(len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store metrics for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    print(f"--- Starting Training on {len(train_dataset)} images ---")
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        # Calculate Averages
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(valid_dataset)
        epoch_train_acc = (train_correct / len(train_dataset)) * 100
        epoch_val_acc = (val_correct / len(valid_dataset)) * 100

        # Save to history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Summary: Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    # Save everything (Weights + History)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'class_names': train_dataset.classes
    }, 'plant_model_complete.pth')
    print("All contents saved to plant_model_complete.pth")

# This 'if' statement is REQUIRED on Windows to prevent the program from crashing/hanging
if __name__ == '__main__':
    train_model()

