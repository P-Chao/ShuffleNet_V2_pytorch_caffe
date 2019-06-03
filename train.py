import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import time
import copy
import shutil
import sys
import matplotlib.pyplot as plt
import pandas as pd

import cv2

from shufflenet_v2 import Network

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.45, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),     # 裁剪到224,224
        #transforms.RandomHorizontalFlip(),     # 随机水平翻转给定的PIL.Image,概率为0.5。即：一半的概率翻转，一半的概率不翻转。
        transforms.ToTensor(),                 # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的FloadTensor
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),     # 裁剪到224,224
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'D:/Data/cnn'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),   # 同时进行transform
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, 
                                             shuffle=True)
                  for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}  # 训练集与验证集数量

class_names = image_datasets['train'].classes  # 样本类别名（子文件夹名）

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # 先深拷贝一份当前模型的参数，后面迭代过程中若遇到更优模型则替换
    best_acc = 0.0   # 初始准确率

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()    # 训练的时候进行学习率规划，其定义在下面给出
                model.train()  # Set model to training mode  设置为训练模式
            else:
                model.eval()  # Set model to evaluate mode  设置为测试模式

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # wrap them in Variable
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                #plt.pause(100)

                #im = torchvision.utils.make_grid(inputs.cpu())
                #print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #print(outputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                save_error_image = True
                if save_error_image:
                    error_idx = np.argwhere(preds.cpu().numpy() != labels.cpu().numpy())
                    img = inputs[0].cpu().numpy().transpose(1, 2, 0)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).to('cpu').numpy()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(running_corrects)
            print(dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:  # 当验证时遇到了更好的模型则予以保留
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())   # 深拷贝模型参数

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)   # 载入最优模型参数
    return model


if __name__=='__main__':
    data_transforms = {
        'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
        'val': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
    }
    data_dir = 'D:/Data/cnn'

    model = Network(2, 1.0)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler)
