#!/usr/bin/env python3

import json, sys, os


def store_json_training(epochs, train_loss):
    """Store train loss of each training epoch to save the results for generation of plots
       without retraining."""
    d = {
        "Train loss" : train_loss,
        "Epoch" : epochs
    }
    f = open('../Plot/train_data.json', 'w')
    json.dump(d, f, indent = 4)


def store_json_offset(offset_output, offset_labels, offset_xy):
    """Store values of last epoch for offset plots."""
    d = {
        "Offset against Output" : offset_output,
        "Offset coordinates" : offset_xy,
        "Offset against Label": offset_labels
    }
    f = open('../Plot/offset_data.json', 'w')
    json.dump(d, f, indent = 4)


def store_json_test(epochs, test_accuracy, f1_score, validation_loss):
    """Store validation loss, accuracy and F1 score of each test epoch for generation of plots
       without retesting on details."""
    d = {
        "Validation loss" : validation_loss,
        "Epoch" : epochs,
        "Accuracy" : test_accuracy,
        "F1-Score" : f1_score
    }
    f = open('../Plot/test_data.json', 'w')
    json.dump(d, f, indent = 4)


def store_json_heatmap(epochs, precision, recall, f1_score):
    """Store precision, recall, F1 score of each test epoch for generation of plots without retesting
       on raw images."""
    d = {
        "Precision" : precision,
        "Epoch" : epochs,
        "Recall" : recall,
        "F1-Score" : f1_score
    }
    f = open('../Plot/heatmap_data.json', 'w')
    json.dump(d, f, indent = 4)


def store_json_thresholddata(thresholds, f1_score, best_threshold, best_f1, precision, recall):
    """Store values of last epoch for threshold plots."""
    d = {
        "List of thresholds" : thresholds,
        "F1-Score" : f1_score,
        "Best threshold" : best_threshold,
        "Best F1-Score" : best_f1,
        "Precision" : precision,
        "Recall" : recall
    }
    f = open('../Plot/threshold_data.json', 'w')
    json.dump(d, f, indent = 4)
