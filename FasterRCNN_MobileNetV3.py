import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn # toto nebude treba, je to na 320ky obrazky
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class FasterRCNN_MobileNetV3:

    size_x = 500.0
    size_y = 500.0

    custom_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((int(size_x), int(size_y))),
        torchvision.transforms.ToTensor()
    ])

    def __init__(self, parameters_file_path, train_set, test_set, val_set, detection_threshold=0.5):
        self.parameters_path = parameters_file_path
        if not os.path.exists(self.parameters_path):
            raise FileNotFoundError(parameters_file_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_set, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        self.model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2, weights=None)
        self.model.to(self.device)
        self.detection_threshold = detection_threshold
        self.early_stopper = EarlyStopper(patience=10, min_delta=0.1)

    def collate_fn(self, batch):
        result = list(batch)
        for i in range(len(batch)):
            # pristup taky ze si dame za sebou x, y, sirka, vyska
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
                points = {'boxes': filtered_bboxes.to(self.device),
                          'labels': torch.ones(filtered_bboxes.shape[0], dtype=torch.long).to(self.device)}
                for j in range(len(points['boxes'])):
                    points['boxes'][j][2:][::4] += points['boxes'][j][0:][::4]  # pretransformovanie vysky sirky na body
                    points['boxes'][j][3:][::4] += points['boxes'][j][1:][::4]
                    # pretransformovanie bboxov na spravnu velkost
                    points['boxes'][j][0:][::2] = points['boxes'][j][0:][::2] / float(batch[i][0].size[0]) * self.size_x
                    points['boxes'][j][1:][::2] = points['boxes'][j][1:][::2] / float(batch[i][0].size[1]) * self.size_y
                    # if points['boxes'][j][2:][::4] - points['boxes'][j][0:][::4] < 1 or points['boxes'][j][3:][::4] - \
                    #         points['boxes'][j][1:][::4] < 1:
                    #     print(points['boxes'][j])
                img = self.custom_transforms(batch[i][0])
                result[i] = list([img, points])
            else:
                img = self.custom_transforms(batch[i][0])
                result[i] = list([img, None])
        return tuple(zip(*result))

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer

    def train_batch(self, img, label, optimizer):
        images = torch.stack(img)
        self.model.train()
        optimizer.zero_grad()
        loss_dict = self.model(images.to(self.device), label)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        losses.backward()
        optimizer.step()
        return loss_value

    @torch.no_grad()
    def val_batch(self, img, label):
        images = torch.stack(img)
        self.model.eval()
        target = []
        preds = []

        outputs = self.model(images.to(self.device))

        # For mAP calculation
        for m in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = label[m]['boxes'].detach().cpu()
            true_dict['labels'] = label[m]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[m]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[m]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[m]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)

        metric = MeanAveragePrecision()
        metric.update(preds, target)

        precision = metric.compute()

        return precision['map']

    def train_model(self, num_of_epochs, use_pretrained_weights=True):
        optimizer = self.get_optimizer()

        if use_pretrained_weights:
            self.model.load_state_dict(torch.load(self.parameters_path, map_location=self.device))

        train_epoch, val_epoch = [], []
        real_epoch_counter = 0
        for epoch in range(num_of_epochs):
            real_epoch_counter += 1
            print('Epoch:', epoch)
            train_batch_losses, val_batch_precisions = [], []
            print('Train loader')
            for img, labels in self.train_loader:
                train_batch_loss = self.train_batch(img, labels, optimizer)
                train_batch_losses.append(train_batch_loss)
            print('Validation loader')
            for img, labels in self.val_loader:
                val_batch_precision = self.val_batch(img, labels)
                val_batch_precisions.append(val_batch_precision)
            train_epoch.append(np.mean(train_batch_losses))
            val_epoch.append(np.mean(val_batch_precisions))
            if self.early_stopper.early_stop(np.mean(val_batch_precisions)):
                print('Stopped because of early stopping on epoch', epoch)
                break

        torch.save(self.model.state_dict(), f=self.parameters_path)

        # Vypis training loss
        plt.subplot(121)
        plt.plot(range(real_epoch_counter), train_epoch, label="train_loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Wider Face model")
        plt.show()
        # Vypis validation precision
        plt.subplot(122)
        plt.plot(range(real_epoch_counter), val_epoch, label="val_precision")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.title("Training Wider Face model")
        plt.show()

    def test_model(self):
        # testovanie modelu
        self.model.load_state_dict(torch.load(self.parameters_path, map_location=self.device))
        for img_batch, label in self.val_loader:
            self.model.eval()
            with torch.no_grad():
                for i in range(len(img_batch)):
                    img = torch.unsqueeze(img_batch[i], dim=0)
                    self.model.eval()
                    outputs = self.model(img.to(self.device))
                    img_out = torch.permute(img.squeeze(), (1, 2, 0))

                    # vyber bounding boxov so skore vacsim ako detection_treshold
                    # get score for all the predicted objects
                    pred_scores = outputs[0]['scores'].cpu()
                    # get all the predicted bounding boxes
                    pred_bboxes = outputs[0]['boxes'].cpu()
                    # get boxes above the threshold score
                    boxes = pred_bboxes[pred_scores >= self.detection_threshold]
                    plt.subplot(121)
                    plt.imshow(img_out)
                    plt.title(" Actual Image ")
                    fig = plt.gcf()
                    ax = fig.add_subplot(122)
                    plt.imshow(img_out)
                    for i in range(boxes.shape[0]):  # vypis bounding boxov
                        x = boxes[i][0]
                        y = boxes[i][1]
                        w = boxes[i][2] - x
                        h = boxes[i][3] - y
                        rect1 = plt.Rectangle((x, y), width=w, height=h, fill=False, edgecolor='red',
                                            linewidth=1, facecolor='none')
                        ax.add_patch(rect1)
                    plt.title(" Actual Image with bounding boxes")
                    plt.show()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_precision = float('-inf')

    def early_stop(self, validation_precision):
        if validation_precision > self.max_validation_precision:
            self.max_validation_precision = validation_precision
            self.counter = 0
        elif validation_precision < (self.max_validation_precision - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
