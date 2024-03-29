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


def test_net(testloader, classes, b_size, net, criterion, number_of_epochs, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score, test_images, test_image_label, test_image_pred):
    """Test on the detail_test set, save example images, and calculate scores
       (loss, accuracy, F1 score, tp, tn, fp, fn)."""
    with torch.no_grad():
        val_loss = 0.0
        val_i = 0
        out = 0.0
        image_count = [6,3]

        for index, sample in enumerate(testloader, 0):
            inputs, labels = sample
            outputs = net(inputs.cuda())
            predicted = outputs > 0.96
            out = '%.6f' % outputs

            if epoch == (number_of_epochs - 1):
                if image_count[int(labels)] > 0:
                    test_images += [inputs[0][0]]
                    test_image_label += [classes[int(labels)]]
                    test_image_pred += [out]
                    image_count[int(labels)] = image_count[int(labels)] - 1
            
            labels = labels.float().reshape(torch.Size(outputs.shape))

            loss = criterion(outputs, labels.cuda())
            
            loss = float(loss.data.cpu().numpy())
            val_loss += loss
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
                    (epoch+1, val_loss/val_i))
    
        if epoch == (number_of_epochs-1):
            print('Test Epoch: %d\n #true positives: {}\n #true negatives: {}\n #false positives: {}\n #false negatives: {}'.format(true_positives, true_negatives, false_positives, false_negatives) % (epoch+1))

    test_accuracy += [correct / total]
    f1_score += [f1]

    print('*** Finished testing in epoch {} ***\n The current accuracy of the network on the test images is %d %%'.format(epoch + 1) % (100 * correct / total))


def loop_epochs(testloader, classes, b_size, net, criterion, number_of_epochs):
    """Prepare needed parameters and test on the detail_test set for each trained epoch."""
    test_accuracy = []
    f1_score = []
    validation_loss = []

    test_images = []
    test_image_label = []
    test_image_pred = []

    save_epoch = 0

    for epoch in range(number_of_epochs):
        save_epoch = epoch

        print('*** Start testing in epoch {} ***'.format(epoch + 1))
        true_positives = 0.0
        true_negatives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        net.load_state_dict(torch.load('../Training/training_epoch-{}'.format(save_epoch)))
        net.eval()
        test_net(testloader, classes, b_size, net, criterion, number_of_epochs, epoch, true_positives, true_negatives, false_positives, false_negatives, validation_loss, test_accuracy, f1_score, test_images, test_image_label, test_image_pred)

    return test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred


def test(testloader, classes, b_size, net, criterion, number_of_epochs):
    test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred = loop_epochs(testloader, classes, b_size, net, criterion, number_of_epochs)
    return test_accuracy, f1_score, validation_loss, test_images, test_image_label, test_image_pred