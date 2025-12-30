import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=10):
    # model = models.mobilenet_v2(weights=None)
    model = models.mobilenet_v2(weights=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)#做一个全连接层
    return model

