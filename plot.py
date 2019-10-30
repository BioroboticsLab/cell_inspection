#!/usr/bin/env python3

import json, sys, os
import datetime
import numpy as np
import scipy.stats
from my_viridis import my_viridis

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import itertools
import scipy.stats


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg(1, 2, 0)))
    plt.axis('off')
    plt.show()


def plot_train_test_graph(number_of_epochs, date_time, epochs, train_loss, test_accuracy, f1_score, validation_loss):
    """Plot train loss of the training set and validation loss, test accuracy and F1 score of the test set
       over epochs to evaluate the model."""
    f, ax = plt.subplots(1, figsize = (10,5))
    plt.xlabel('epochs')
    plt.ylabel('train loss, validation loss, test accuracy, F1 score')
    ax.plot(epochs, train_loss, 'r', label = 'Train loss')
    ax.plot(epochs, validation_loss, 'b', label = 'Validation loss')
    ax.plot(epochs, test_accuracy, 'g', label = '(Test) Accuracy / 100')
    ax.plot(epochs, f1_score, 'm', label = 'F1 Score')
    plt.axis([0, (number_of_epochs - 1), 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, number_of_epochs, (number_of_epochs // 10)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc = 'center left', frameon = True)
    leg.get_frame().set_edgecolor('k')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('../Plot/train_test_graph_{}.svg'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def plot_heatgraph(number_of_epochs, epochs, train_loss, precision, recall, f1_score, date_time):
    """Plot train loss of the training set and precision recall and F1 score of the raw images
       over epochs to evaluate the model."""
    f, ax = plt.subplots(1, figsize = (10,5))
    plt.xlabel('epochs')
    plt.ylabel('train loss, precision, recall, F1 score')
    ax.plot(epochs, train_loss, 'r', label = 'Train loss')
    ax.plot(epochs, precision, 'g', label = 'Precision')
    ax.plot(epochs, recall, 'b', label = 'Recall')
    ax.plot(epochs, f1_score, 'm', label = 'F1 score')
    plt.axis([0, (number_of_epochs - 1), 0, 1])
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, number_of_epochs, (number_of_epochs // 10)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc = 'center left', frameon = True)
    leg.get_frame().set_edgecolor('k')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('../Plot/heatgraph_{}.svg'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)


def plot_test_images(date_time, test_images, test_image_label, test_image_pred):
    """Plot examples of 6 notbee details and 3 bee details with their corresponding labels
       and the prediction of the network."""
    fig, axarr = plt.subplots(3,3, figsize = (3,3))
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
    plt.savefig('../Plot/test_images_{}.png'.format(date_time.strftime('%d-%m-%y_%H:%M')), bbox_inches='tight', dpi=100)


def plot_offset_output(date_time, offset_labels, offset_output, offset_xy, sigma):
    """Plot desired and actual output of the network over length of the input offset
       in the last epoch of the training. Also plot offset distribution."""
    x_axis_out = []
    y_axis_out = []
    
    # replaced by label_curve, just used for verification
    # x_axis_lab = []
    # y_axis_lab = []
    
    offset_x = []
    offset_y = []

    for x,y in offset_output:
        x_axis_out += [x]
        y_axis_out += [y]
        
    # for x,y in offset_labels:
    #     x_axis_lab += [x]
    #     y_axis_lab += [y]

    for x,y in offset_xy:
        offset_x += [x]
        offset_y += [y]
    
    sigma=sigma
    probability_distribution = scipy.stats.norm(scale=sigma)
    probability_at_zero = probability_distribution.pdf(0.0)
    pdf_x = np.linspace(0, 27, 271)
    label_curve = probability_distribution.pdf(pdf_x) / probability_at_zero

    plt.subplot(2,1,1)
    plt.xlabel('offset-norm')
    plt.ylabel('label and output')
    plt.plot(pdf_x, label_curve, color='r', label='Label')
    # plt.scatter(x_axis_lab, y_axis_lab, color='b', marker='.', label='Labelstrength')
    plt.scatter(x_axis_out, y_axis_out, color='g', marker='+', label='Output')
    plt.axis([0, 27, 0, 1])
    plt.tick_params(axis='x', direction='inout')
    plt.tick_params(axis='y', direction='inout')
    leg = plt.legend(bbox_to_anchor=(1.01, 0.5), loc = 'center left', frameon = True)
    leg.get_frame().set_edgecolor('k')
    plt.grid(False)
    plt.tight_layout()

    plt.subplot(2,1,2)
    plt.axis('equal')
    plt.xlabel('offset_x')
    plt.ylabel('offset_y')
    plt.scatter(offset_x, offset_y, color='b', marker='.', label='Offset coordinates')
    plt.axis([-15, 15, -15, 15])
    plt.tick_params(axis='x', direction='inout')
    plt.tick_params(axis='y', direction='inout')
    leg = plt.legend(bbox_to_anchor=(1.01, 0.5), loc = 'center left', frameon = True)
    leg.get_frame().set_edgecolor('k')
    plt.grid(False)
    plt.tight_layout()

    plt.savefig('../Plot/offset_graph_{}.svg'.format(date_time.strftime('%d-%m-%y_%H:%M')), dpi = 100)

    
def plot_outputs(img_name, epoch, original_image, output_greater, date_time):
    """Plot output of the network."""
    fig, ax =  plt.subplots(figsize=(20, 10))
    ax.imshow(output_greater, cmap='gray')
    plt.axis('off')
    plt.savefig('../Heatmap/{}_output_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)

    
def plot_nonmax(img_name, epoch, original_image, image_max, date_time):
    """Plot results of non-maximum suppression."""
    fig, ax =  plt.subplots(figsize=(20, 10))
    ax.imshow(image_max, cmap='gray')
    plt.axis('off')
    plt.savefig('../Heatmap/{}nonmax_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_label_max_coordinates(img_name, epoch, original_image, list_label, list_max, date_time):
    """Plot raw image with ground truth markers and predicted positions found in the last epoch of
       testing to evaluate the model."""
    fig, ax =  plt.subplots(figsize=(20, 10))
    ax.imshow(original_image, cmap='gray')
    ax.scatter(list_label[:, 0], list_label[:, 1], marker='+')
    ax.scatter(list_max[:, 0], list_max[:, 1], marker='.')
    plt.axis('off')
    plt.savefig('../Heatmap/{}_label_max_coordinates_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_transparent_heatmap_overlay(img_name, epoch, original_image, output_greater, list_label, list_max, date_time):
    """Plot heatmap overlaid onto raw image to visualize spatially resolved predictions of the network"""
    plt.register_cmap(name='VIRIDIS', data=my_viridis)
    cmap = plt.get_cmap('VIRIDIS')

    width_origimg = original_image.shape[1]
    height_origimg = original_image.shape[0]
    print('{}x{}'.format(width_origimg, height_origimg))
    width_out = output_greater.shape[1]
    height_out = output_greater.shape[0]
    print('{}x{}'.format(width_out, height_out))

    extent = [0, width_origimg, height_origimg, 0]

    fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(original_image, cmap='gray', extent=extent)
    heatmap = ax.imshow(output_greater, interpolation='bicubic', extent=extent, cmap=cmap)
    trans_heatmap = mtransforms.Affine2D().scale((4000-54)/4000, (3000-54)/3000).translate(27,27) + ax.transData
    heatmap.set_transform(trans_heatmap)
    ax.axis('off')
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel(None, rotation=-90, va='bottom')
    plt.savefig('../Heatmap/{}_heatmap_overlay_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_label_max_coordinates_with_transparent_heatmap_overlay(img_name, epoch, original_image, output_greater, list_label, list_max, date_time):
    """Plot raw image with heatmap, ground truth markers and predicted positions found in the last epoch
       of testing."""
    plt.register_cmap(name='VIRIDIS', data=my_viridis)
    cmap = plt.get_cmap('VIRIDIS')

    width_origimg = original_image.shape[1]
    height_origimg = original_image.shape[0]
    print('{}x{}'.format(width_origimg, height_origimg))
    width_out = output_greater.shape[1]
    height_out = output_greater.shape[0]
    print('{}x{}'.format(width_out, height_out))

    extent = [0, width_origimg, height_origimg, 0]

    fig, ax =  plt.subplots(figsize=(20, 10))
    ax.imshow(original_image, cmap="gray", extent=extent)
    ax.scatter(list_label[:, 0], list_label[:, 1], marker="+")
    ax.scatter(list_max[:, 0], list_max[:, 1], marker=".")
    heatmap = ax.imshow(output_greater, interpolation='bicubic', extent=extent, cmap=cmap)
    trans_heatmap = mtransforms.Affine2D().scale((4000-54)/4000, (3000-54)/3000).translate(27,27) + ax.transData
    heatmap.set_transform(trans_heatmap)
    ax.axis('off')
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel(None, rotation=-90, va='bottom')
    plt.savefig('../Heatmap/{}_coordinates_heatmap_overlay_{}_{}'.format(img_name, date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)