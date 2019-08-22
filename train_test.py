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


def test_net(testloader, classes, b_size, net, criterion, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score, test_images, test_image_label, test_image_pred):
    with torch.no_grad():
        val_loss = 0.0
        val_i = 0
        out = 0.0

        for index, sample in enumerate(testloader, 0):
            inputs, labels = sample
            outputs = net(inputs)
            predicted = outputs > 0.5
            out = '%.6f' % outputs

            test_images += [inputs[0][0]]
            test_image_label += [classes[labels[0]]]
            test_image_pred += [out]

            labels = labels.float()

            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_i = index + 1

            true_positives += int(predicted) and int(labels)
            true_negatives += not(int(predicted)) and not(int(labels))
            false_positives += int(predicted) and not(int(labels))
            false_negatives += not(int(predicted)) and int(labels)

            total = true_positives + true_negatives + false_positives + false_negatives
            correct = true_positives + true_negatives

            print('Predicted: {}\t Solution: {}'.format(int(predicted), int(labels)))
            print('output: %.6f' % float(outputs))

               
        validation_loss += [val_loss / val_i]

        if (false_positives == 0 and true_positives == 0):
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        if (false_negatives == 0 and true_positives == 0):
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        numerator = precision * recall
        denominator = precision + recall
        if (denominator == 0):
            f1 = 0
        else:
            f1 = 2 * (numerator / denominator)

        print('Test Epoch: %d\t Average validation loss: %.6f\n Precision: {}\t Recall: {}'.format(precision, recall) %
                    (epoch + 1, val_loss / val_i))
        
    test_accuracy += [correct / total]
    f1_score += [f1]

    print('*** Finished testing in epoch {} ***\n The current accuracy of the network on the test images is %d %%'.format(epoch + 1) % (100 * correct / total))



def loop_epoches(trainloader, testloader, classes, b_size, net, criterion, optimizer, number_of_epoches):
    epoches = []
    train_loss = []

    test_accuracy = []
    f1_score = []
    validation_loss = []

    train_images = []
    train_image_label = []

    test_images = []
    test_image_label = []
    test_image_pred = []

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

        print('*** Start testing in epoch {} ***'.format(epoch + 1))
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0     

        net.eval()
        test_net(testloader, classes, b_size, net, criterion, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score, test_images, test_image_label, test_image_pred)

    return net, epoches, train_loss, train_images, train_image_label, test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred


def train_test(trainloader, testloader, classes, b_size, net, criterion, optimizer, number_of_epoches):
    net_trained, epoches, train_loss, train_images, train_image_label, test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred = loop_epoches(trainloader, testloader, classes, b_size, net, criterion, optimizer, number_of_epoches)
    return epoches, train_loss, test_accuracy, f1_score, validation_loss, train_images, train_image_label, test_images, test_image_label, test_image_pred
