# importing relevant packages

import torch
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import gdown
import torchvision.transforms as transforms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = False, transform= transforms.ToTensor())

custom_transforms = torchvision.transforms.Compose([ #spravi transformaciu img na rovnaku velkost
    torchvision.transforms.Resize((70, 70)),
    # torchvision.transforms.RandomCrop((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = False, transform= custom_transforms)
val_set = torchvision.datasets.WIDERFace('WiderFace_data',split='val', download = False, transform= custom_transforms)
test_set = torchvision.datasets.WIDERFace('WiderFace_data',split='test', download = False, transform= custom_transforms)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

# print(len(train_set))
# print(len(val_set))
# print(len(test_set))

print(train_set[1][0].shape) #zistenie shape dat

model = models.vgg16(weights=None)

print(model) #vypise info o modeli


# getting the optimizer and loss_function

def get_essentials():
  loss_fun = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  return loss_fun, optimizer


epochs = 50
loss_fun, optimizer = get_essentials()


# training and validation loops

train_epoch, val_epoch = [], []
for epoch in range(epochs):
  train_batch_losses, val_batch_losses = [], []
  for data in train_loader:
    train_batch_loss = train_batch(data, model, loss_fun, optimizer)
    train_batch_losses.append(train_batch_loss)
  for data in test_dataloader:
    val_batch_loss = val_batch(data, model, loss_fun, optimizer)
    val_batch_losses.append(val_batch_loss)
  train_epoch.append(np.mean(train_batch_losses))
  val_epoch.append(np.mean(val_batch_losses))


