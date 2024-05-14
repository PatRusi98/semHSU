# importing relevant packages

import torchvision.datasets

import SSD300_VGG16 as vgg16_model
import FasterRCNN_MobileNetV3 as faster_rcnn_model


parameters_path = "C:/Users/patri/PycharmProjects/semHSU/model_parameters.pth" #PC banter
parameters_path_faster = "C:/Users/patri/PycharmProjects/semHSU/model_parameters_faster.pth" #PC banter
# parameters_path = "C:/D/Desktop/School/4.Rocnik/Hlboke_strojove_ucenie/Semestralka/semHSU/model_parameters.pth"

train_set = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = True)
val_set = torchvision.datasets.WIDERFace('WiderFace_data',split='val', download = True)
test_set = torchvision.datasets.WIDERFace('WiderFace_data',split='test', download = True)

ssd300_vgg16 = vgg16_model.SSD300_VGG16(parameters_path, train_set, test_set, val_set)
fasterrcnn_mobilenetv3 = faster_rcnn_model.FasterRCNN_MobileNetV3(parameters_path_faster, train_set, test_set, val_set)

#ssd300_vgg16.train_model(50, False) #True - pokracuje v trenovani nacitanim ulozenych vah, false prepise vahy a zacina prakticky z nuly
#fasterrcnn_mobilenetv3.train_model(10, False) #True - pokracuje v trenovani nacitanim ulozenych vah, false prepise vahy a zacina prakticky z nuly
# ssd300_vgg16.test_model() #testovanie
fasterrcnn_mobilenetv3.test_model() #testovanie
