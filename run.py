#!/usr/bin/env python3

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

import beeimagefolder

import train, test, train_test, heatmap, heatmap_solo #, continue


def plot_train_test_graph(date_time, epoches, train_loss, test_accuracy, f1_score, validation_loss):
    f, ax = plt.subplots(1, figsize = (12,6))
    plt.xlabel('Epoch-Counter')
    ax.plot(epoches, train_loss, 'r', label = 'Training loss')
    ax.plot(epoches, test_accuracy, 'g', label = '(Test) Accuracy / 100')
    ax.plot(epoches, f1_score, 'm', label = 'F1-Score')
    ax.plot(epoches, validation_loss, 'b', label = 'Validation loss')
    plt.axis([0, 39, 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, 40, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc = 'center right', frameon = True)
    plt.grid(True)
    plt.savefig('../Plot/train_test_graph_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def plot_train_graph(date_time, epoches, train_loss):
    f, ax = plt.subplots(1, figsize = (20,6))
    plt.xlabel('Epoch-Counter')
    ax.plot(epoches, train_loss, 'r', label = 'Training loss')
    plt.axis([0, 69, 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, 70, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc = 'center right', frameon = True)
    plt.grid(True)
    plt.savefig('../Plot/train_graph_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def plot_heatgraph(epoches, train_loss, precision, recall, f1_score, date_time):
    f, ax = plt.subplots(1, figsize = (12,6))
    plt.xlabel('Epoch-Counter')
    ax.plot(epoches, train_loss, 'r', label = 'Training loss')
    ax.plot(epoches, f1_score, 'm', label = 'F1-Score')
    ax.plot(epoches, precision, 'g', label = 'Precision')
    ax.plot(epoches, recall, 'b', label = 'Recall')
    plt.axis([0, 39, 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, 40, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc = 'center right', frameon = True)
    plt.grid(True)
    plt.savefig('../Plot/heatgraph_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg(1, 2, 0)))
    plt.axis('off')
    plt.show()


def plot_train_images(date_time, train_images, train_image_label):
    fig, axarr = plt.subplots(4,4, figsize = (3,3))
    for ax, img, label in zip(itertools.chain(*axarr), train_images, train_image_label):
        ax.imshow(img, cmap = 'gray')
        ax.set_title(label, fontsize = 8)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(wspace = 0)
        plt.subplots_adjust(hspace=0.5)
    plt.savefig('../Plot/train_images_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), bbox_inches='tight')


def plot_test_images(date_time, test_images, test_image_label, test_image_pred):
    fig, axarr = plt.subplots(4,4, figsize = (3,3))
    for ax, img, label, pred in zip(itertools.chain(*axarr), test_images, test_image_label, test_image_pred):
        ax.imshow(img, cmap = 'gray')
        ax.set_title(label, fontsize = 7)
        ax.set_xlabel('pred: '+pred, fontsize = 5)
        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        ax.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(wspace = 0.5)
        plt.subplots_adjust(hspace=1.5)
    plt.savefig('../Plot/test_images_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), bbox_inches='tight')


def load_data():
    #
    # Load the data
    # The detail folders contains a 'bee' folder with 78x78 grayscale PNG images of bees in honeycombs
    # and a 'notbee' folder with negative examples.
    # detail_training = ~70-80% of all details, detail_test = ~20-30% of all details
    #
    transform_train = transforms.Compose([
        transforms.RandomOrder([
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness = 0.5, contrast = 0.5)
        ]),
        transforms.CenterCrop(54),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor()
    ])

    transform_test =  transforms.Compose([
        transforms.CenterCrop(54),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor()
    ])

    b_size = 100
    trainset = beeimagefolder.BeeImageFolder(root='../Videos/detail_training', valid_classes=['0 notbee', '1 bee'], transform=transform_train)
    testset = beeimagefolder.BeeImageFolder(root='../Videos/detail_test', valid_classes=['0 notbee', '1 bee'], transform=transform_test)

    print('class_to_idx trainset: {}\t class_to_idx testset: {}'.format(trainset.class_to_idx, testset.class_to_idx))
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
    # shuffle testset ~ get random image-examples
    testloader = torch.utils.data.DataLoader(testset, shuffle=True)

    classes = ('notbee', 'bee')

    return trainloader, testloader, classes, b_size


class Net(nn.Module):
    #
    # Define a Convolutional Neural Network
    # 32x32 -> 30@14x14 -> 60@12x12 -> 60@5x5 -> 120@3x3 -> 120@1x1 -> 1@1x1
    #
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
    trainloader, testloader, classes, b_size = load_data()
    net = Net()
    criterion = nn.BCELoss()
    # optim.Adam learning rate ~ default: 0.001
    optimizer = optim.Adam(net.parameters())
    date_time = datetime.datetime.now()
    number_of_epoches = 80

    if len(args) < 2:
        print("usage: run.py (train|test|train_test|continue|heatmap)")
    
    elif args[1] == 'train':
        epoches, train_loss, train_images, train_image_label = train.train(trainloader, classes, b_size, net, criterion, optimizer, number_of_epoches)
        plot_train_graph(date_time, epoches, train_loss)
        plot_train_images(date_time, train_images, train_image_label)

    elif args[1] == 'test':
        j = json.load(open('../Training/trainloss+epoches.json'))
        train_loss = j["Training loss"]
        epoches = j["Epoch"]
        test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred = test.test(testloader, classes, b_size, net, criterion, number_of_epoches)
        plot_train_test_graph(date_time, epoches, train_loss, test_accuracy, f1_score, validation_loss)
        plot_test_images(date_time, test_images, test_image_label, test_image_pred)
    
    elif args[1] == 'train_test':
        epoches, train_loss, test_accuracy, f1_score, validation_loss, train_images, train_image_label, test_images, test_image_label, test_image_pred = train_test.train_test(trainloader, testloader, classes, b_size, net, criterion, optimizer, number_of_epoches)
        plot_train_test_graph(date_time, epoches, train_loss, test_accuracy, f1_score, validation_loss)
        plot_train_images(date_time, train_images, train_image_label)
        plot_test_images(date_time, test_images, test_image_label, test_image_pred)

    elif args[1] == 'continue':
        #TODO: continue.py
        print('continue.continue() is not yet executable')
    
    elif args[1] == 'heatmap':
        j = json.load(open('../Training/trainloss+epoches.json'))
        train_loss = j["Training loss"]
        epoches = j["Epoch"]
        print (epoches)
        precision, recall, f1_score = heatmap.heatmap(net, date_time, number_of_epoches)
        plot_heatgraph(epoches, train_loss, precision, recall, f1_score, date_time)

    elif args[1] == 'heatmap_solo':
        j = json.load(open('../Training/trainloss+epoches.json'))
        prec, rec = heatmap_solo.heatmap_solo(net, date_time)

    else:
        print('unknown command: {}'.format(args[1]))

if __name__ == '__main__':
    main(sys.argv)