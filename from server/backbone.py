from network.resnet import *
from network.mobilenet_v2 import *
from network.resnet_cifar import resnet18_cifar,resnet50_cifar,resnet34_cifar,resnet101_cifar
import torch.nn as nn
import torch.nn.functional as F
from network.head import *

backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet18_cifar': resnet18_cifar,
    'resnet34_cifar': resnet34_cifar,
    'resnet50_cifar': resnet50_cifar,
    'resnet101_cifar': resnet101_cifar,
    'mobilenet_v2':mobilenet_v2
}

dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet18_cifar': 512,
    'resnet34_cifar': 512,
    'resnet50_cifar': 2048,
    'resnet101_cifar': 2048,
    'mobilenet_v2':1280

}


