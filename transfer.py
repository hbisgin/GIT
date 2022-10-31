import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
import numpy as np
import torchvision

from torchvision import datasets, models, transforms


device = 'cpu'

model  = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2) #two classes
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

#scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

'''for epoch in range(100):
    train() #optimizer.step()
    evaluate() #
    scheduler.step() 

'''

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs = 20)


#################ALternatively
model  = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2) #two classes
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

#scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs = 20)

