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


def train_net(trainloader, classes, b_size, net, criterion, optimizer, epoch, epochs, running_loss, t_loss, train_i, train_loss, save_epoch, number_of_epochs, offset_output, offset_labels, offset_xy):
    """Train on the detail_training set. Save resulting data for further use (testing, plots)"""
    save_epoch_data = open('../Training/training_epoch-{}'.format(save_epoch), 'a')
    save_epoch_data.close()

    for index, batch in enumerate(trainloader, 0):
        inputs, labels = batch
        # index = running counter
        # inputs: batch of up to 200 32x32 tensors
        # labels: batch of up to 200 1x4 tensors
        
        optimizer.zero_grad()
        outputs = net(inputs.cuda())
        
        # labels.size() = torch.Size([200,4])
        # to get labels (in the dataset known as labelstrength), offset_norm, offset_x, offset_y,
        # we need to extract the associated columns. This is done with "select"
        offset_norm = labels.select(1, 1)
        offset_x = labels.select(1, 2)
        offset_y = labels.select(1, 3)
        labels = labels.select(1, 0)
                
        # The offset is only needed for the bee-labels, which can bee extracted from the batch.
        # Since not all batches have the same size (if there is not enough data, the last one will
        # not have the given batchsize), range(inputs.size()[0]) is used instead of range(b_size).
        # According to the labelstrength, labels will not be 1 for every bee-image which is why
        # all labels greater 0 are taken.
        if save_epoch == (number_of_epochs - 1):
            for i in range(inputs.size()[0]):
                if labels[i].item() > 0:
                    offset_output += [(offset_norm[i].item(), outputs[i].item())]
                    offset_labels += [(offset_norm[i].item(), labels[i].item())]
                    offset_xy += [(offset_x[i].item(), offset_y[i].item())]

        labels = labels.float().reshape(torch.Size(outputs.shape))
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        loss = float(loss.data.cpu().numpy())
        running_loss += loss
        t_loss += loss
        train_i = index + 1

        if index % 10 == 9:
            print('Train Epoch: %d\t Index: %3d\t Loss %.6f' %
                    (epoch + 1, index + 1, running_loss / 10))
            running_loss = 0.0
    print(train_i)
        
    train_loss += [t_loss / train_i]

    print('Train Epoch: %d\t Average loss: %.6f' %
                    (epoch + 1, t_loss / train_i))

    torch.save(net.state_dict(), '../Training/training_epoch-{}'.format(save_epoch))

    print('*** Finished training in epoch {} ***'.format(epoch + 1))


def loop_epochs(trainloader, classes, b_size, net, criterion, optimizer, number_of_epochs):
    """Prepare needed parameters and train on the detail_training set for a given number of epochs."""
    epochs = []
    train_loss = []

    offset_output = []
    offset_labels = []
    offset_xy = []

    save_epoch = 0

    for epoch in range(number_of_epochs):
        running_loss = 0.0
        t_loss = 0.0
        train_i = 0
        save_epoch = epoch

        epochs += [epoch]

        print('*** Start training in epoch {} ***'.format(epoch + 1))
        net.train()
        train_net(trainloader, classes, b_size, net, criterion, optimizer, epoch, epochs, running_loss, t_loss, train_i, train_loss, save_epoch, number_of_epochs, offset_output, offset_labels, offset_xy)    

    return net, epochs, train_loss, offset_output, offset_labels, offset_xy


def train(trainloader, classes, b_size, net, criterion, optimizer, number_of_epochs):
    net_trained, epochs, train_loss, offset_output, offset_labels, offset_xy = loop_epochs(trainloader, classes, b_size, net, criterion, optimizer, number_of_epochs)
    return epochs, train_loss, offset_output, offset_labels, offset_xy
