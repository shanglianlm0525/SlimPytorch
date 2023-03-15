# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/3/15 8:55
# @Author : liumin
# @File : utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys
import time
import torchvision
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def prepare_model(model_type='resnet18', pretrained=True, num_classes=2):
    if model_type=='resnet50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type=='resnet34':
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type=='resnet101':
        model = resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type=='resnet152':
        model = resnet152(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def prepare_data(data_dir='/home/lmin/data/hymenoptera', img_size=(224, 224), train_batch_size=16, eval_batch_size=1, num_workers=8, only_eval=False):

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    if only_eval:
        return eval_loader

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, eval_loader


def eval_model(model, eval_loader, device):
    model.eval()
    model.to(device)

    running_corrects = 0
    for inputs, labels in eval_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
    eval_acc = running_corrects / len(eval_loader.dataset)
    return eval_acc


def train_model(model, train_loader, eval_loader, device):
    num_epochs = 100
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer_ft.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        lr_scheduler_ft.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # deep copy the model
        eval_acc = eval_model(model=model, eval_loader=eval_loader, device=device)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'eval Acc: {eval_acc:.4f} Best Acc: {best_acc:.4f}')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model