import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from torch.cuda import amp
import time
from PIL import Image
import random

# Mixup data augmentation
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Customize CIFAR-10 dataset
class CIFAR10Custom(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        if self.train:
            for i in range(1, 6):
                batch_path = os.path.join(root, f'data_batch_{i}')
                with open(batch_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.targets += batch[b'labels']
            self.data = np.concatenate(self.data)
        else:
            batch_path = os.path.join(root, 'test_batch')
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data = batch[b'data']
                self.targets = batch[b'labels']
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # Random cropping
    transforms.RandomHorizontalFlip(),       # Random horizontal flipping
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),  # Normalization
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

data_root = '../cifar-10-batches-py'  #------------------------------------------------- dataset path

# Create dataloader
trainset = CIFAR10Custom(root=data_root, train=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = CIFAR10Custom(root=data_root, train=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Wide ResNet
def get_wide_resnet50_2(num_classes=10):
    model = models.wide_resnet50_2(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

model = get_wide_resnet50_2(num_classes=10)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training and validation functions
def train_epoch(model, dataloader, criterion, optimizer, scaler, mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
            with amp.autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        if mixup:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() +
                        (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Model training
num_epochs = 200
best_acc = 0.0
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
scaler = amp.GradScaler()

# Create training_log
log_file = '../Output/training_log.txt'
with open(log_file, 'w') as f:
    f.write(f"Training Log\n{'='*30}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Model: Wide ResNet50-2\n")
    f.write(f"Optimizer: AdamW\n")
    f.write(f"Learning Rate: 0.001\n")
    f.write(f"Weight Decay: 5e-4\n")
    f.write(f"Number of Epochs: {num_epochs}\n\n")

start_time = time.time()

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, scaler, mixup=True)
    val_loss, val_acc = validate(model, testloader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step()

    with open(log_file, 'a') as f:
        f.write(f'Epoch [{epoch}/{num_epochs}] '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), '../Weights/best_model.pth')

    # Early stop condition (optional)
    if best_acc >= 95.0:
        with open(log_file, 'a') as f:
            f.write("stop training early\n")
        break

end_time = time.time()
with open(log_file, 'a') as f:
    f.write(f'Training completed in {end_time - start_time:.2f} seconds.\n')
    f.write(f'Best Validation Accuracy: {best_acc:.2f}%\n')
