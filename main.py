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

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import numpy as np

parameters_path = "C:/Users/patri/PycharmProjects/semHSU/model_parameters.pth"
# parameters_path = "C:/D/Desktop/School/4.Rocnik/Hlboke_strojove_ucenie/Semestralka/semHSU/model_parameters.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
dropout = 0.5
detection_threshold = 0.5
size_x = 512.0
size_y = 512.0

custom_transforms = torchvision.transforms.Compose([ #spravi transformaciu img
    torchvision.transforms.Resize((int(size_x), int(size_y))),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #toto teraz netreba pre SSD
])

def test_transf(batch):
    return torch.tensor(batch['bbox'])

def collate_fn(batch):
    result = list(batch)
    for i in range(len(batch)):
        #pristup taky ze si dame za sebou x, y, sirka, vyska
        if batch[i][1] is not None:
            batch[i][1]['bbox'] = torch.stack(sorted(batch[i][1]['bbox'], key=lambda bbox: bbox[2], reverse=True))
            filtered_bboxes = list()
            for bbox in batch[i][1]['bbox']:
                if bbox[2] > 1 and bbox[3] > 1:
                    filtered_bboxes.append(bbox.float())
            if len(filtered_bboxes) != 0:
                filtered_bboxes = torch.stack(filtered_bboxes)
            else:
                filtered_bboxes = torch.empty((0, 4))
            points = {'boxes': filtered_bboxes.to(device), 'labels': torch.ones(filtered_bboxes.shape[0], dtype=torch.long).to(device)}
            for j in range(len(points['boxes'])):
                points['boxes'][j][2:][::4] += points['boxes'][j][0:][::4] #pretransformovanie vysky sirky na body
                points['boxes'][j][3:][::4] += points['boxes'][j][1:][::4]
                #pretransformovanie bboxov na spravnu velkost
                points['boxes'][j][0:][::2] = points['boxes'][j][0:][::2] / float(batch[i][0].size[0]) * size_x
                points['boxes'][j][1:][::2] = points['boxes'][j][1:][::2] / float(batch[i][0].size[1]) * size_y
                # points['boxes'][j][0] = float(points['boxes'][j][0]) / float(batch[i][0].size[0]) * size_x
                # points['boxes'][j][2] = float(points['boxes'][j][2]) / float(batch[i][0].size[0]) * size_x
                # points['boxes'][j][1] = float(points['boxes'][j][1]) / float(batch[i][0].size[1]) * size_y
                # points['boxes'][j][3] = float(points['boxes'][j][3]) / float(batch[i][0].size[1]) * size_y
                if points['boxes'][j][2:][::4] - points['boxes'][j][0:][::4] < 1 or points['boxes'][j][3:][::4] - points['boxes'][j][1:][::4] < 1:
                    print(points['boxes'][j])
            # points = torch.tensor(points)
            # points['boxes'] = F.pad(input=points['boxes'], pad=(0, (136 - points['boxes'].shape[0])), mode='constant', value=0) #pre spravny shape
            # testik = torch.tensor(batch[i][0].size).repeat(68)
            # points_out = points_out / testik # podelime shapom povodneho obrazku
            img = custom_transforms(batch[i][0])
            result[i] = list([img, points])
        else:
            img = custom_transforms(batch[i][0])
            result[i] = list([img, None])
    return tuple(zip(*result))


train_set = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = True)
val_set = torchvision.datasets.WIDERFace('WiderFace_data',split='val', download = True)
test_set = torchvision.datasets.WIDERFace('WiderFace_data',split='test', download = True)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn) #collate_fn tam musi byt lebo tie anotacie to dava stale rozny shape a to collate to zabali
test_loader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

# print(len(train_set))
# print(len(val_set))
# print(len(test_set))

# print(train_set[0][0].shape) #zistenie shape dat

model = models.detection.ssd300_vgg16(num_classes=2, pretrained=False)  # for training

# testujeme model upravu
# model = models.detection.ssd300_vgg16() #pretreined for testing

# print(model) #vypise info o modeli

model = model.to(device)

# getting the optimizer and loss_function

# def get_essentials():
#     #loss_fun = nn.L1Loss()
#     loss_fun = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     return loss_fun, optimizer
#
#
# def train_batch(img, label, model, loss_fun, optimizer):
#     images = torch.stack(img)
#     model.train()
#     optimizer.zero_grad()
#     loss_dict = model(images.to(device), label) #vlozim do modelu cely batch
#
#     losses = sum(loss for loss in loss_dict.values())
#     loss_value = losses.item()
#
#     losses.backward()
#     optimizer.step()
#     return loss_value
#
# @torch.no_grad()
# def val_batch(img, label, model, loss_fun, optimizer):
#     images = torch.stack(img)
#     model.eval()
#     target = []
#     preds = []
#
#     outputs = model(images.to(device))
#
#     # For mAP calculation using Torchmetrics.
#     #####################################
#     for m in range(len(images)):
#         true_dict = dict()
#         preds_dict = dict()
#         true_dict['boxes'] = label[m]['boxes'].detach().cpu()
#         true_dict['labels'] = label[m]['labels'].detach().cpu()
#         preds_dict['boxes'] = outputs[m]['boxes'].detach().cpu()
#         preds_dict['scores'] = outputs[m]['scores'].detach().cpu()
#         preds_dict['labels'] = outputs[m]['labels'].detach().cpu()
#         preds.append(preds_dict)
#         target.append(true_dict)
#     #####################################
#
#     metric = MeanAveragePrecision()
#     metric.update(preds, target)
#
#     # Compute the results
#     precision = metric.compute()
#
#     return precision['map']
#
#
# epochs = 30
# loss_fun, optimizer = get_essentials()
#
# #Trening a validacia
# # # training and validation loops
#
# train_epoch, val_epoch = [], []
# for epoch in range(epochs):
#     print('Epoch: ', epoch)
#     train_batch_losses, val_batch_losses = [], []
#     print('Train loader')
#     for img, labels in train_loader:
#         train_batch_loss = train_batch(img, labels, model, loss_fun, optimizer)
#         train_batch_losses.append(train_batch_loss)
#     print('Validation loader')
#     for img, labels in val_loader:
#         val_batch_loss = val_batch(img, labels, model, loss_fun, optimizer)
#         val_batch_losses.append(val_batch_loss)
#     train_epoch.append(np.mean(train_batch_losses))
#     val_epoch.append(np.mean(val_batch_losses))
#     #print('Train: ' + np.mean(train_batch_losses))
#     #print('Validation: ' + np.mean(val_batch_losses)) #tu uz by mal byt precision
#
# #save model parameters
# torch.save(model.state_dict(), f=parameters_path)
#
# plt.plot(range(epochs), train_epoch, label="train_loss")
# plt.plot(range(epochs), val_epoch, label="val_loss")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training Wider Face model")
# plt.show()

#testovanie modelu
model.load_state_dict(torch.load(parameters_path, map_location=device))
for img_batch, label in val_loader:
    model.eval()
    with torch.no_grad():
        for i in range(len(img_batch)):
            img = torch.unsqueeze(img_batch[i], dim=0)
            model.eval()
            pred_points = model(img.to(device))
            # labels = torch.stack(label)
            # loss_val = loss_fun(pred_points.cpu(), labels)
            metric = MeanAveragePrecision(iou_type="bbox")

            # Update metric with predictions and respective ground truth
            testik = list([label[i]])
            metric.update(pred_points, testik)

            # Compute the results
            precision = metric.compute()

            # img = torch.unsqueeze(img_out, dim=0)
            # outputs = model(img.to(device))
            img_out = torch.permute(img.squeeze(), (1, 2, 0))
            # # get score for all the predicted objects
            # pred_scores = outputs[0]['scores'].detach().cpu().numpy()
            # # get all the predicted bounding boxes
            # pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
            # # get boxes above the threshold score
            # boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
            plt.subplot(121)
            plt.imshow(img_out)
            plt.title(" Actual Image ")
            fig = plt.gcf()
            ax = fig.add_subplot(122)
            plt.imshow(img_out)
            patches = []
            for i in range(pred_points[0]['boxes'].shape[0]): #vypis bounding boxov
                x = pred_points[0]['boxes'][i][0].cpu()
                y = pred_points[0]['boxes'][i][1].cpu()
                w = pred_points[0]['boxes'][i][2].cpu() - x
                h = pred_points[0]['boxes'][i][3].cpu() - y
                rect1 = plt.Rectangle((x, y), width=w, height=h, fill=False, edgecolor='red',
                                             linewidth=1, facecolor='none')
                ax.add_patch(rect1)
            plt.title(" Actual Image with keyponts")
            plt.show()

