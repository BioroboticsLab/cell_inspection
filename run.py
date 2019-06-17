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

import train, test, train_test #, continue


def plot_graph(date_time, epoches, train_loss, test_accuracy, f1_score, validation_loss):
    # TODO: best test-score, learning rate
    # TODO: Seperate Loss and Accuracy
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
    plt.savefig('../Plot/graph_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


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
    # The detail folder contains a 'bee' folder with 78x78 grayscale PNG images of bees in honeycombs
    # and a 'notbee' folder with negative examples.
    #
    full_dataset = torchvision.datasets.ImageFolder(root = "../Videos/detail")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    trainset.dataset = copy(trainset.dataset)

    trainset.dataset.transform = transforms.Compose([
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

    testset.dataset.transform = transforms.Compose([
        transforms.CenterCrop(54),
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor()
    ])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset)

    classes = ('notbee', 'bee')

    return trainloader, testloader, classes


class Net(nn.Module):
    #
    # Define a Convolutional Neural Network
    # 32x32 -> 30@14x14 -> 60@12x12 -> 60@5x5 -> 120@3x3 -> 120@1x1 -> 1@1x1
    #
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5, stride = 2)
       # self.drop1 = nn.Dropout2d(p = 0.1)
        self.conv2 = nn.Conv2d(30, 60, 3, stride = 1)
       # self.norm1 = nn.BatchNorm2d(60)
       # self.drop2 = nn.Dropout2d(p = 0.1)
        self.conv3 = nn.Conv2d(60, 60, 3, stride = 2)
       # self.norm2 = nn.BatchNorm2d(60)
       # self.drop3 = nn.Dropout2d(p = 0.1)
        self.conv4 = nn.Conv2d(60, 120, 3, stride = 1)
       # self.norm3 = nn.BatchNorm2d(120)
       # self.drop4 = nn.Dropout2d(p = 0.1)
        self.conv5 = nn.Conv2d(120, 120, 3, stride = 1)
       # self.drop5 = nn.Dropout()
        self.conv6 = nn.Conv2d(120, 1, 1)

    def forward(self, x):
       # x = self.drop1(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
       # x = self.drop2(self.norm1(F.relu(self.conv2(x))))
        x = F.relu(self.conv2(x))
       # x = self.drop3(self.norm2(F.relu(self.conv3(x))))
        x = F.relu(self.conv3(x))
       # x = self.drop4(self.norm3(F.relu(self.conv4(x))))
        x = F.relu(self.conv4(x))
       # x = self.drop5(F.relu(self.conv5(x)))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))

        return x


def main(args):
    trainloader, testloader, classes = load_data()
    net = Net()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    date_time = datetime.datetime.now()

    if len(args) < 2:
        print("usage: run.py (train|test|train_test|continue)")
    
    elif args[1] == 'train':
        train_images, train_image_label = train.train(trainloader, classes, net, criterion, optimizer, date_time)
        plot_train_images(date_time, train_images, train_image_label)

    elif args[1] == 'test':
        j = json.load(open('../Training/trainloss+epoches.json'))
        train_loss = j["Training loss"]
        epoches = j["Epoch"]
        test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred = test.test(testloader, classes, net, criterion, date_time)
        plot_graph(date_time, epoches, train_loss, test_accuracy, f1_score, validation_loss)
        plot_test_images(date_time, test_images, test_image_label, test_image_pred)
    
    elif args[1] == 'train_test':
        epoches, train_loss, test_accuracy, f1_score, validation_loss, train_images, train_image_label, test_images, test_image_label, test_image_pred = train_test.train_test(trainloader, testloader, classes, net, criterion, optimizer, date_time)
        plot_graph(date_time, epoches, train_loss, test_accuracy, f1_score, validation_loss)
        plot_train_images(date_time, train_images, train_image_label)
        plot_test_images(date_time, test_images, test_image_label, test_image_pred)

    elif args[1] == 'continue':
        #TODO: continue.py
        print('continue.continue() is not yet executable')
    
    else:
        print('unknown command: {}'.format(args[1]))

if __name__ == '__main__':
    main(sys.argv)