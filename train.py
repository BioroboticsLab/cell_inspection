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


def plot_train_graph(date_time, epoches, train_loss):
    f, ax = plt.subplots(1, figsize = (12,6))
    plt.xlabel('Epoch-Counter')
    ax.plot(epoches, train_loss, 'r', label = 'Training loss')
    plt.axis([0, 39, 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, 40, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc = 'center right', frameon = True)
    plt.grid(True)
    plt.savefig('../Plot/train_graph_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def train_net(trainloader, classes, net, criterion, optimizer, epoch, epoches, running_loss, t_loss, train_i, train_loss, train_images, train_image_label, save_epoch):
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
    #TODO: stop using average loss ~ instead: every 10 batches or all

    print('Train Epoch: %d\t Average loss: %.6f' %
                    (epoch + 1, t_loss / train_i))

    torch.save(net.state_dict(), '../Training/training_epoch-{}'.format(save_epoch))

    print('*** Finished training in epoch {} ***'.format(epoch + 1))


def loop_epoches(trainloader, classes, net, criterion, optimizer):
    epoches = []
    train_loss = [] 

    train_images = []
    train_image_label = []

    save_epoch = 0

    for epoch in range(40):
        running_loss = 0.0
        t_loss = 0.0
        train_i = 0
        save_epoch = epoch

        epoches += [epoch]

        print('*** Start training in epoch {} ***'.format(epoch + 1))
        net.train()
        train_net(trainloader, classes, net, criterion, optimizer, epoch, epoches, running_loss, t_loss, train_i, train_loss, train_images, train_image_label, save_epoch)
        store_json(train_loss, epoches)
    

    return net, epoches, train_loss, train_images, train_image_label


def train(trainloader, classes, net, criterion, optimizer, date_time):
    net_trained, epoches, train_loss, train_images, train_image_label = loop_epoches(trainloader, classes, net, criterion, optimizer)
    plot_train_graph(date_time, epoches, train_loss)
    return train_images, train_image_label