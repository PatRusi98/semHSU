# importing relevant packages

import torch
import torchvision.datasets
from matplotlib.patches import Rectangle
from torchvision import transforms # !NEMAZAT toto treba aj ked sa tvari ze nie
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

import semHSU.SSD300_VGG16 as vgg16_model

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import numpy as np

# parameters_path = "C:/Users/patri/PycharmProjects/semHSU/model_parameters.pth"
parameters_path = "C:/D/Desktop/School/4.Rocnik/Hlboke_strojove_ucenie/Semestralka/semHSU/model_parameters.pth"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# dropout = 0.5
# detection_threshold = 0.5
# size_x = 512.0
# size_y = 512.0
#
# custom_transforms = torchvision.transforms.Compose([ #spravi transformaciu img
#     torchvision.transforms.Resize((int(size_x), int(size_y))),
#     torchvision.transforms.ToTensor()
#     # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #toto teraz netreba pre SSD
# ])
#
# def test_transf(batch):
#     return torch.tensor(batch['bbox'])
#
# def collate_fn(batch):
#     result = list(batch)
#     for i in range(len(batch)):
#         #pristup taky ze si dame za sebou x, y, sirka, vyska
#         if batch[i][1] is not None:
#             batch[i][1]['bbox'] = torch.stack(sorted(batch[i][1]['bbox'], key=lambda bbox: bbox[2], reverse=True))
#             filtered_bboxes = list()
#             for bbox in batch[i][1]['bbox']:
#                 if bbox[2] > 1 and bbox[3] > 1:
#                     filtered_bboxes.append(bbox.float())
#             if len(filtered_bboxes) != 0:
#                 filtered_bboxes = torch.stack(filtered_bboxes)
#             else:
#                 filtered_bboxes = torch.empty((0, 4))
#             points = {'boxes': filtered_bboxes.to(device), 'labels': torch.ones(filtered_bboxes.shape[0], dtype=torch.long).to(device)}
#             for j in range(len(points['boxes'])):
#                 points['boxes'][j][2:][::4] += points['boxes'][j][0:][::4] #pretransformovanie vysky sirky na body
#                 points['boxes'][j][3:][::4] += points['boxes'][j][1:][::4]
#                 #pretransformovanie bboxov na spravnu velkost
#                 points['boxes'][j][0:][::2] = points['boxes'][j][0:][::2] / float(batch[i][0].size[0]) * size_x
#                 points['boxes'][j][1:][::2] = points['boxes'][j][1:][::2] / float(batch[i][0].size[1]) * size_y
#                 # points['boxes'][j][0] = float(points['boxes'][j][0]) / float(batch[i][0].size[0]) * size_x
#                 # points['boxes'][j][2] = float(points['boxes'][j][2]) / float(batch[i][0].size[0]) * size_x
#                 # points['boxes'][j][1] = float(points['boxes'][j][1]) / float(batch[i][0].size[1]) * size_y
#                 # points['boxes'][j][3] = float(points['boxes'][j][3]) / float(batch[i][0].size[1]) * size_y
#                 if points['boxes'][j][2:][::4] - points['boxes'][j][0:][::4] < 1 or points['boxes'][j][3:][::4] - points['boxes'][j][1:][::4] < 1:
#                     print(points['boxes'][j])
#             # points = torch.tensor(points)
#             # points['boxes'] = F.pad(input=points['boxes'], pad=(0, (136 - points['boxes'].shape[0])), mode='constant', value=0) #pre spravny shape
#             # testik = torch.tensor(batch[i][0].size).repeat(68)
#             # points_out = points_out / testik # podelime shapom povodneho obrazku
#             img = custom_transforms(batch[i][0])
#             result[i] = list([img, points])
#         else:
#             img = custom_transforms(batch[i][0])
#             result[i] = list([img, None])
#     return tuple(zip(*result))


train_set = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = True)
val_set = torchvision.datasets.WIDERFace('WiderFace_data',split='val', download = True)
test_set = torchvision.datasets.WIDERFace('WiderFace_data',split='test', download = True)

ssd300_vgg16 = vgg16_model.SSD300_VGG16(parameters_path, train_set, test_set, val_set)

ssd300_vgg16.test_model()

