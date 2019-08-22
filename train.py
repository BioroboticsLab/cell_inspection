import json, sys
import random
import datetime
from copy import copy

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import itertools


def store_json(train_loss, epoches):
    d = {
        "Training loss" : train_loss,
        "Epoch" : epoches
    }
    f = open('../Training/trainloss+epoches.json', 'w')
    json.dump(d, f, indent = 4)


def train_net(trainloader, classes, b_size, net, criterion, optimizer, epoch, epoches, running_loss, t_loss, train_i, train_loss, train_images, train_image_label, save_epoch):
    save_epoch_data = open('../Training/training_epoch-{}'.format(save_epoch), 'a')
    save_epoch_data.close()

    for index, batch in enumerate(trainloader, 0):
        inputs, labels = batch
        # index = running counter
        # inputs: 32x32 tensor
        # labels: 1x1

        train_images += [inputs[0][0]]
        train_image_label += [classes[labels[0]]]

        labels = labels.float()

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        t_loss += loss.item()
        train_i = index + 1

        if index % 10 == 9:
            print('Train Epoch: %d\t Index: %3d\t Loss %.6f' %
                    (epoch + 1, index + 1, running_loss / 10))
            running_loss = 0.0
        
    train_loss += [t_loss / train_i]

    print('Train Epoch: %d\t Average loss: %.6f' %
                    (epoch + 1, t_loss / train_i))

    torch.save(net.state_dict(), '../Training/training_epoch-{}'.format(save_epoch))

    print('*** Finished training in epoch {} ***'.format(epoch + 1))


def loop_epoches(trainloader, classes, b_size, net, criterion, optimizer, number_of_epoches):
    epoches = []
    train_loss = [] 

    train_images = []
    train_image_label = []

    save_epoch = 0

    for epoch in range(number_of_epoches):
        running_loss = 0.0
        t_loss = 0.0
        train_i = 0
        save_epoch = epoch

        epoches += [epoch]

        print('*** Start training in epoch {} ***'.format(epoch + 1))
        net.train()
        train_net(trainloader, classes, b_size, net, criterion, optimizer, epoch, epoches, running_loss, t_loss, train_i, train_loss, train_images, train_image_label, save_epoch)
        store_json(train_loss, epoches)
    

    return net, epoches, train_loss, train_images, train_image_label


def train(trainloader, classes, b_size, net, criterion, optimizer, number_of_epoches):
    net_trained, epoches, train_loss, train_images, train_image_label = loop_epoches(trainloader, classes, b_size, net, criterion, optimizer, number_of_epoches)
    return epoches, train_loss, train_images, train_image_label