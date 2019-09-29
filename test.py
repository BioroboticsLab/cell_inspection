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


def test_net(testloader, classes, b_size, net, criterion, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score, test_images, test_image_label, test_image_pred, offset_output, offset_xy):
    with torch.no_grad():
        val_loss = 0.0
        val_i = 0
        out = 0.0

        for index, sample in enumerate(testloader, 0):
            inputs, labels = sample
            outputs = net(inputs)
            predicted = outputs > 0.8
            out = '%.6f' % outputs

            test_images += [inputs[0][0]]
            test_image_label += [classes[int(labels)]]
            test_image_pred += [out]
            
            labels = labels.float().reshape(torch.Size(outputs.shape))

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


def loop_epoches(testloader, classes, b_size, net, criterion, number_of_epoches):
    test_accuracy = []
    f1_score = []
    validation_loss = []

    offset_output = []
    offset_xy = []


    test_images = []
    test_image_label = []
    test_image_pred = []

    save_epoch = 0

    for epoch in range(number_of_epoches):
        save_epoch = epoch

        print('*** Start testing in epoch {} ***'.format(epoch + 1))
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0     

        net.load_state_dict(torch.load('../Training/training_epoch-{}'.format(save_epoch)))
        net.eval()
        test_net(testloader, classes, b_size, net, criterion, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score, test_images, test_image_label, test_image_pred, offset_output, offset_xy)

    return test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred #, offset_output, offset_xy


def test(testloader, classes, b_size, net, criterion, number_of_epoches):
    test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred = loop_epoches(testloader, classes, b_size, net, criterion, number_of_epoches)
    return test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred #, offset_output, offset_xy