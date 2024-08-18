#!/usr/bin/env python
# coding: utf-8

# ##Flowers 102 dataset

# Installing all the necessary files to then import into the notebook

# In[ ]:

import subprocess
import sys


subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])



# Importing all the necessary imports

# In[2]:


####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.io
import random
import skimage.io as skio
import os
import glob
import json
from tqdm import tqdm
import torch
from torchviz import make_dot
import torchvision
from torch import nn
from torch import optim
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast





# Changing device to GPU if available

# In[ ]:


####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# Load in the data from the 102 Flowers dataset

# In[ ]:


####
train_dataset = datasets.Flowers102(
    root="data",
    split="train",
    download =True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.Flowers102(
    root="data",
    split="test",
    download =True
)

val_dataset = datasets.Flowers102(
    root="data",
    split="val",
    download =True
)


# Imports the names of the flowers from a JSON file

# In[ ]:


####
with open('flower_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

print(len(flower_to_name))
flower_to_name


# Splits up the names

# In[6]:


####
category_map = sorted(flower_to_name.items(), key=lambda x: int(x[0]))
category_names = [cat[1] for cat in category_map]


# Removes the spaces and replaces it with an "_"

# In[7]:


####
codes = np.array([name.replace(" ", "_") for name in category_names])


# In[8]:


####
img_labels_mat = scipy.io.loadmat('data/flowers-102/imagelabels.mat')
img_labels = img_labels_mat.get('labels')
targets = img_labels[0]


# Prints a graph of all the flowers and how many photos there are of each flower species

# In[ ]:


####
unique_labels, counts = np.unique(targets, return_counts=True)
label_names = codes[unique_labels-1]
label_counts = dict(zip(label_names, counts))


names = list(label_counts.keys())
values = list(label_counts.values())

plt.figure(figsize=(18, 8))
plt.bar(names, values)
plt.xlabel('Flower Categories')
plt.ylabel('Number of Images')
plt.title('Number of Images per Flower Category')
plt.xticks(rotation=90)
plt.show(block=False)
plt.pause(0.1)


# Show random 8 images from the dataset with its labels

# In[ ]:


####
image_nums = random.sample(range(len(train_dataset)), 8)
fig, axes = plt.subplots(2, 4, figsize=(15, 6))

for i, ax in zip(image_nums, axes.flatten()):

    image, label = train_dataset[i]
    image_np = image.permute(1, 2, 0).numpy()

    flower_name = codes[label]

    ax.imshow(image_np)
    ax.set_title(f'{flower_name}, {label}')
    ax.axis('off')

plt.show(block=False)
plt.pause(0.1)


# #Transform the data to ensure it fits by finding out the mean and STD of the data
# 
# *   Resize image so all the same size
# *   Put a pixel square in the centre
# *   Convert image to Tensor Flow
# *   Normalize pixel values
# 

# In[11]:


####
transform = transforms.Compose([
    transforms.Resize(226),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])

dataset = datasets.Flowers102(
    root="data",
    download =True,
    transform = transform

)


# In[ ]:


####
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std


mean, std = compute_mean_std(dataset)
print(f"Dataset mean: {mean}, Dataset std: {std}")


# Transforming the data to increase the size of the training dataset to improve the model

# In[13]:


####
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(mean, std)
])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# In[14]:


####
train_dataset = datasets.Flowers102(
    root="data",
    split="train",
    download =True,
    transform=train_transform
)

test_dataset = datasets.Flowers102(
    root="data",
    split="test",
    download =True,
    transform=transform
)

val_dataset = datasets.Flowers102(
    root="data",
    split="val",
    download =True,
    transform=transform
)


# Use DataLoader to load the data

# In[15]:


####
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


# #Show the pre-processed images

# Shows the transformed data but not using PIL

# In[ ]:


####
for x, y in train_loader:
    x = x.to(device)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))

    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            image_data = x[idx].cpu().permute(1, 2, 0)
            label = y[idx].item()
            flower_name = flower_to_name.get(str(label)).replace(" ", "_")
            ax[i, j].imshow(image_data)
            ax[i, j].axis('off')
            ax[i, j].set_title(f'{flower_name}, Label: {label}')

    plt.show(block=False)
    plt.pause(0.1)
    break


# Shows images that have been processed and transformed using PIL Image

# In[ ]:


####
def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis('off')
fig, axes = plt.subplots(1, 8, figsize=(15, 2.5))
dataiter = iter(train_loader)
images, labels = next(dataiter)

for i in range(8):
    image = images[i]
    label = labels[i].item()
    pil_img = transforms.ToPILImage()(image)
    flower_name = codes[label]


    axes[i].imshow(pil_img)
    axes[i].set_title(f'{flower_name.replace(" ", "_")}\n(Label: {label})')
    axes[i].axis('off')

plt.show(block=False)
plt.pause(0.1)


# ##Make CNN Model

# In[ ]:


####
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Dropout(0.35),

            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((14, 14))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.6),
            nn.Linear(1024, 102)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, 512 * 14 * 14)
        x = self.classifier(x)
        return x

model = CNN()
print(model)


# #Use model on testing data

# Load the model up and put in evaluation mode

# In[19]:


####
model = CNN()


# In[ ]:


####
model.load_state_dict(torch.load('model_flowers102.pth', map_location=device))


# In[ ]:


####
model = model.to(device)


model.eval()


total = 0
correct = 0
train_accuracies = []

with torch.no_grad():
    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)


        total += labels.size(0)
        correct += (pred == labels).sum().item()

train_accuracy = (correct / total) * 100
train_accuracies.append(train_accuracy)

print(f'Training Accuracy: {train_accuracy}%')


# In[ ]:


####
model = model.to(device)

model.eval()


total = 0
correct = 0
val_accuracies = []



with torch.no_grad():
    for images, labels in val_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)


        total += labels.size(0)
        correct += (pred == labels).sum().item()


val_accuracy = (correct / total) * 100


val_accuracies.append(val_accuracy)

print(f'Validation Accuracy: {val_accuracy}%')


# In[ ]:


####
model = model.to(device)


model.eval()


total = 0
correct = 0
test_accuracies = []

with torch.no_grad():
    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)


        total += labels.size(0)
        correct += (pred == labels).sum().item()

test_accuracy = (correct / total) * 100

test_accuracies.append(test_accuracy)

print(f'Testing Accuracy: {test_accuracies}%')

