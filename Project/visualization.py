import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image

# Customize CIFAR-10 dataset
class CIFAR10Custom(Dataset):
    def __init__(self, root, train=False, transform=None):
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
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 class name
cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

data_root = '../cifar-10-batches-py'  #-----------------------------------------Dataset path

# Create dataset and dataloader
testset = CIFAR10Custom(root=data_root, train=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model structure
def get_wide_resnet50_2(num_classes=10):
    model = models.wide_resnet50_2(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Model initialization and loading
model = get_wide_resnet50_2(num_classes=10)
checkpoint_path = '../Weights/best_model.pth'  #-----------------------------------Weight path
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f'successfully loading model weight from {checkpoint_path}')
else:
    print(f'cannot find the checkpoint file {checkpoint_path}, please check the path')
model = model.to(device)
model.eval()

# Define inverse normalization function for converting images back to visualization format
def denormalize(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Randomly display 10 images and their predicted results
num_images = 10
fig = plt.figure(figsize=(15, 10))
for i, (inputs, targets) in enumerate(testloader):
    if i >= num_images:
        break
    inputs = inputs.to(device)
    targets = targets.numpy()[0]
    outputs = model(inputs)
    _, predicted = outputs.max(1)
    predicted = predicted.cpu().numpy()[0]

    img = denormalize(inputs.cpu().data[0])

    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    ax.imshow(img)
    title_color = 'green' if predicted == targets else 'red'
    title_text = f'real: {cifar10_classes[targets]}\nprediction: {cifar10_classes[predicted]}'
    ax.set_title(title_text, color=title_color)
plt.tight_layout()
plt.show()

