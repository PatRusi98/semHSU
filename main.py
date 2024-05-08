# importing relevant packages

import torchvision.datasets


import semHSU.SSD300_VGG16 as vgg16_model


# parameters_path = "C:/Users/patri/PycharmProjects/semHSU/model_parameters.pth"
parameters_path = "C:/D/Desktop/School/4.Rocnik/Hlboke_strojove_ucenie/Semestralka/semHSU/model_parameters.pth"

train_set = torchvision.datasets.WIDERFace('WiderFace_data',split='train', download = True)
val_set = torchvision.datasets.WIDERFace('WiderFace_data',split='val', download = True)
test_set = torchvision.datasets.WIDERFace('WiderFace_data',split='test', download = True)

ssd300_vgg16 = vgg16_model.SSD300_VGG16(parameters_path, train_set, test_set, val_set)

ssd300_vgg16.test_model()

