# importing relevant packages

import torch
import torchvision.datasets
from torchvision import transforms # !NEMAZAT toto treba aj ked sa tvari ze nie
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.5

# data = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = False, transform= transforms.ToTensor())

custom_transforms = torchvision.transforms.Compose([ #spravi transformaciu img na rovnaku velkost
    torchvision.transforms.Resize((224, 224)),
    # torchvision.transforms.RandomCrop((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def test_transf(batch):
    return torch.tensor(batch['bbox'])

def collate_fn(batch):
    result = list(batch)
    for i in range(len(batch)):
        # print(batch[i][1]['bbox'].flatten())
        # print(batch[i][1]['bbox'].flatten()[1:][::2])
        x_axis_points = batch[i][1]['bbox'].flatten()[0:][::2] #x suradnice
        y_axis_points = batch[i][1]['bbox'].flatten()[1:][::2] #y suradnice
        x_axis_points = list(x_axis_points / batch[i][0].size[0])
        y_axis_points = list(y_axis_points / batch[i][0].size[1])

        points = x_axis_points + y_axis_points
        points = torch.tensor(points)
        points_out = F.pad(input=points, pad=(0, (136 - points.shape[0])), mode='constant', value=0) #pre spravny shape
        img = custom_transforms(batch[i][0])
        result[i] = list([img, points_out])

    return tuple(zip(*result))


train_set = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = True) #chcelo by to tu transform toho dictionary
val_set = torchvision.datasets.WIDERFace('WiderFace_data',split='val', download = True, transform= custom_transforms)
test_set = torchvision.datasets.WIDERFace('WiderFace_data',split='test', download = True, transform= custom_transforms)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn) #collate_fn tam musi byt lebo tie anotacie to dava stale rozny shape a to collate to zabali
test_loader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

# print(len(train_set))
# print(len(val_set))
# print(len(test_set))

# print(train_set[0][0].shape) #zistenie shape dat

model = models.vgg16(weights=None)

# print(model) #vypise info o modeli pred zmenou

final_predictor = nn.Sequential(
    # nn.Linear(25088, 128, bias=True),
    # nn.ReLU(inplace=True),
    # nn.Dropout(p=dropout, inplace=False),
    # nn.Linear(128, 64, bias=True),
    # nn.ReLU(inplace=True),
    # nn.Dropout(p=dropout, inplace=False),
    # nn.Linear(64, 32, bias=True),
    # nn.ReLU(),
    # nn.Dropout(p=dropout, inplace=False),
    # nn.Linear(32, 4, bias=True),
    #test odtial
    nn.Linear(25088, 300, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout, inplace=False),
    nn.Linear(300, 136, bias=True)
)

model.classifier = final_predictor

# print(model) #vypise info o modeli po zmene

model = model.to(device)

# getting the optimizer and loss_function

def get_essentials():
  loss_fun = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  return loss_fun, optimizer


def train_batch(img, label, model, loss_fun, optimizer):
  images = torch.stack(img)
  model.train()
  pred_points = model(images) #vlozim do modelu cely batch
  # print(startX, startY, endX, endY)
  # print(pred_points[0]) #testovaci print tu by asi chcelo si vypisat tie body ze co to vlastne je alebo jak
  labels = torch.stack(label)
  loss_val = loss_fun(pred_points, labels) #vystup pred_points bude mat druhy rozmer 1000 tak musim enastavit aj labels
  loss_val.backward()
  optimizer.step()
  optimizer.zero_grad()
  return loss_val.item()

@torch.no_grad()
def val_batch(img, labels, model, loss_fun, optimizer):
  model.eval()
  pred_points = model(img)
  loss_val = loss_fun(pred_points, labels)
  return loss_val.item()


epochs = 50
loss_fun, optimizer = get_essentials()

# training and validation loops

train_epoch, val_epoch = [], []
for epoch in range(epochs):
  train_batch_losses, val_batch_losses = [], []
  for img, labels in train_loader:
    train_batch_loss = train_batch(img, labels, model, loss_fun, optimizer)
    train_batch_losses.append(train_batch_loss)
  for img, labels in test_loader:
    val_batch_loss = val_batch(img, labels, model, loss_fun, optimizer)
    val_batch_losses.append(val_batch_loss)
  train_epoch.append(np.mean(train_batch_losses))
  val_epoch.append(np.mean(val_batch_losses))


plt.plot(range(epochs), train_epoch, label="train_loss")
plt.plot(range(epochs), val_epoch, label="test_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Facial Keypoints model")
plt.show()

