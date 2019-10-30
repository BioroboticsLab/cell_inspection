#!/usr/bin/env python3

import json, sys, os
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

import beeimagefolder, mydataset

import train, test, heatmap, find_best_threshold, plot, store


def load_data():
    """Load the data: The detail folders contain a '1 bee' folder with 78x78 grayscale PNG images
       of bees in honeycombs and a '0 notbee' folder with negative examples."""
    b_size = 200
    sigma = 15

    transform_train = transforms.Compose([
        transforms.RandomOrder([
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5)
        ]),
        transforms.CenterCrop(54),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(54),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    trainset = beeimagefolder.BeeImageFolder(root='../Videos/detail_training', valid_classes=['0 notbee', '1 bee'])
    translated_trainset = mydataset.MyDataset(dataset=trainset, sigma=sigma, transform=transform_train)
    testset = beeimagefolder.BeeImageFolder(root='../Videos/detail_test', valid_classes=['0 notbee', '1 bee'], transform=transform_test)

    print('class_to_idx trainset: {}\t class_to_idx testset: {}'.format(trainset.class_to_idx, testset.class_to_idx))

    trainloader = torch.utils.data.DataLoader(translated_trainset, num_workers=2, batch_size=b_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset)

    classes = ('notbee', 'bee')

    return trainloader, testloader, classes, b_size, sigma


class Net(nn.Module):
    """Define a Convolutional Neural Network:
       32x32 -> 30@14x14 -> 60@12x12 -> 60@5x5 -> 120@3x3 -> 120@1x1 -> 1@1x1"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5, stride = 2)
        self.conv2 = nn.Conv2d(30, 60, 3, stride = 1)
        self.conv3 = nn.Conv2d(60, 60, 3, stride = 2)
        self.conv4 = nn.Conv2d(60, 120, 3, stride = 1)
        self.conv5 = nn.Conv2d(120, 120, 3, stride = 1)
        self.conv6 = nn.Conv2d(120, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        
        return x


def main(args):
    """Prepare needed parameters and call the function selected via command line."""
    trainloader, testloader, classes, b_size, sigma = load_data()
    net = Net()
    net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.Adam(net.parameters())
    date_time = datetime.datetime.now()
    number_of_epochs = 300
    
    if len(args) < 2:
        print('usage: run.py (train|test|heatmap|threshold|plot_all)')
    
    elif args[1] == 'train':
        epochs, train_loss, offset_output, offset_labels, offset_xy = train.train(trainloader, classes, b_size, net, criterion, optimizer, number_of_epochs)
        store.store_json_training(epochs, train_loss)
        store.store_json_offset(offset_output, offset_labels, offset_xy)     
        plot.plot_offset_output(date_time, offset_labels, offset_output, offset_xy, sigma)

    elif args[1] == 'test':
        j = json.load(open('../Plot/train_data.json'))
        train_loss = j["Train loss"]
        epochs = j["Epoch"]
        test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred = test.test(testloader, classes, b_size, net, criterion, number_of_epochs)
        store.store_json_test(epochs, test_accuracy, f1_score, validation_loss)
        plot.plot_train_test_graph(number_of_epochs, date_time, epochs, train_loss, test_accuracy, f1_score, validation_loss)
        plot.plot_test_images(date_time, test_images, test_image_label, test_image_pred)
        
    elif args[1] == 'heatmap':
        if len(args) < 3:
            print('usage: run.py heatmap <json-file(s)>')
        else:
            json_train = json.load(open('../Plot/train_data.json'))
            train_loss = json_train["Train loss"]
            epochs = json_train["Epoch"]
            given_arguments = args[2:]
            precision, recall, f1_score = heatmap.heatmap(given_arguments, net, date_time, number_of_epochs)
            store.store_json_heatmap(epochs, precision, recall, f1_score)
            plot.plot_heatgraph(number_of_epochs, epochs, train_loss, precision, recall, f1_score, date_time)

    elif args[1] == 'threshold':
        if len(args) < 3:
            print('usage: run.py threshold <json-file(s)>')
        else:
            given_arguments = args[2:]
            thresholds = np.arange(0.01, 1, 0.01)
            thresholds = thresholds.tolist()
            precision, recall, f1_score, best_threshold, best_f1 = find_best_threshold.find_best_threshold(given_arguments, net, date_time, number_of_epochs, thresholds)
            store.store_json_thresholddata(thresholds, f1_score, best_threshold, best_f1, precision, recall)
        
    else:
        print('unknown command: {}'.format(args[1]))

if __name__ == '__main__':
    main(sys.argv)