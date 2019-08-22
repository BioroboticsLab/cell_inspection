import torch
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from run import *
from torchvision import transforms
import skimage.io, skimage.transform
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from scipy.spatial.kdtree import KDTree
from my_viridis import my_viridis

def get_orig_position(liste):
    # only add half of the steps ~ kernel = 5, add 2 instead of 4 (height + 2, width + 2)
    liste[:, 0] = (np.floor(((liste[:, 0] + 0.5 + 1 + 1) * (883/442) + 1 + 1) * 2 + 2 - 0.5)) * (54/32)
    liste[:, 1] = (np.floor(((liste[:, 1] + 0.5 + 1 + 1) * (1179/590) + 1 + 1) * 2 + 2 - 0.5)) * (54/32)
    return liste


def create_score(epoch, list_label, list_max, true_positives, false_positives, false_negatives, precision, recall, f1_score):
    i = 0
    while i < len(list_label):
        target = list_label[i]
        print(target)
        max_distance = 27
        distances = np.linalg.norm(list_max - target, axis=1)
        distances[distances > max_distance] = np.inf
        best_match = np.argmin(distances)
        match_distance = distances[best_match]
        if np.isinf(match_distance):
            false_negatives += 1
        else:
            true_positives += 1
            list_max[best_match] = np.inf
        print('best_match: {}\t match_distance: {}'.format(best_match, match_distance))
        i = i + 1
    false_positives = len(list_max) - true_positives
    prec = true_positives / (true_positives + false_positives)
    rec = true_positives / (true_positives + false_negatives)
    numerator = prec * rec
    denominator = prec + rec
    f1 = 2 * (numerator / denominator)

    print('Epoch: %d\t Precision: {}\t Recall: {}'.format(prec, rec) % (epoch))
    
    precision += [prec]
    recall += [rec]
    f1_score += [f1]


def plot_label_max_coordinates(epoch, original_image, list_label, list_max, date_time):
    fig, ax =  plt.subplots(figsize=(20, 10))
    ax.imshow(original_image, cmap='gray')
    ax.scatter(list_label[:, 0], list_label[:, 1], marker='+')
    ax.scatter(list_max[:, 0], list_max[:, 1], marker='.')
    plt.axis('off')
    plt.savefig('../Heatmap/label_max_coordinates_{}_{}'.format(date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_transparent_heatmap_overlay(epoch, original_image, output_greater, list_label, list_max, date_time):
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

    plt.savefig('../Heatmap/heatmap_overlay_{}_{}'.format(date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_label_max_coordinates_with_transparent_heatmap_overlay(epoch, original_image, output_greater, list_label, list_max, date_time):
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

    plt.savefig('../Heatmap/coordinates_heatmap_overlay_{}_{}'.format(date_time.strftime('%d-%m-%y_%H:%M'), epoch), bbox_inches='tight')
    plt.close(fig)


def plot_heatmap_maxfilter_mask_overlay(epoch, image_max, output_greater, mask, net_coord, date_time):
    fig, ax = plt.subplots(figsize=(20,10))

    ax.imshow(image_max, cmap='gray')
    ax.imshow(output_greater, alpha=0.5, cmap='autumn')
    ax.imshow(mask, alpha=0.5, cmap='cool')
    ax.scatter(net_coord[:,0], net_coord[:,1], marker='x')
    ax.axis('off')

    plt.savefig('../Heatmap/{}_heatmap_maxfilter_mask_overlay_{}'.format(epoch, date_time.strftime('%d-%m-%y_%H:%M')), bbox_inches='tight')
    plt.close(fig)


def plot_coordinates_next_to_mask_overlay(epoch, original_image, list_label, list_max, image_max, output_greater, mask, net_coord, date_time):
    #TODO
    print('function not yet executable!')



def create_heatmaps(net, epoch, true_positives, false_positives, false_negatives, precision, recall, f1_score, date_time):
    im = skimage.io.imread('test.png')
    im = rgb2gray(im)
    im = im[:,:]

    original_image = im.copy()
    print(im.shape, im.dtype, im.max())
    im = skimage.transform.rescale(im, scale=(32 / 54))
    im = im.reshape((1, 1, im.shape[0], im.shape[1]))
    im = im.astype(np.float32)
    net.eval()
    print(im.shape)
    with torch.no_grad():
        output = net(torch.tensor(im))
        predicted = output > 0.8
        print('output.shape: ', output.shape)
        print('output: {}\t predicted: {}'.format(output, predicted))

    output_greater = output[0, 0].numpy().copy()

    image_max = ndi.maximum_filter(output_greater.copy(), size=8, mode='constant')
    mask = output_greater == image_max
    mask &= output_greater > 0.8

    net_coord = np.nonzero(mask)
    net_coord = np.fliplr(np.column_stack(net_coord))

    list_max = get_orig_position(net_coord.copy())

    j = json.load(open('test_markiert.json'))
    red_pixels = j["red"]
    blue_pixels = j["blue"]
    
    list_label = red_pixels + blue_pixels
    list_label.sort()
    list_label = np.array(list_label)
    
    print('*** list_max (maxima of net_coord) ***\n {}\n *** list_label (marked bees) ***\n {}'.format(list_max, list_label))

    # compare points of list_label to points of list_max
    plot_label_max_coordinates(epoch, original_image, list_label, list_max, date_time)

    # heatmap above original image
    plot_transparent_heatmap_overlay(epoch, original_image, output_greater, list_label, list_max, date_time)

    # list_max and list_label coordinates above heatmap
    plot_label_max_coordinates_with_transparent_heatmap_overlay(epoch, original_image, output_greater, list_label, list_max, date_time)

    # heatmap with maxfilter and mask (heatmap == maxfilter)
    plot_heatmap_maxfilter_mask_overlay(epoch, image_max, output_greater, mask, net_coord, date_time)

    # heatmap with maxfilter and mask next to coordinates
    plot_coordinates_next_to_mask_overlay(epoch, original_image, list_label, list_max, image_max, output_greater, mask, net_coord, date_time)

    create_score(epoch, list_label, list_max, true_positives, false_positives, false_negatives, precision, recall, f1_score)


def loop_epoches(net, date_time, number_of_epoches):
    f1_score = []
    precision = []
    recall = []

    save_epoch = 0

    for epoch in range(number_of_epoches):
        save_epoch = epoch
        print('*** Create heatmaps using state of epoch {} ***'.format(epoch))
        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        net.load_state_dict(torch.load('../Training/training_epoch-{}'.format(save_epoch)))
        create_heatmaps(net, epoch, true_positives, false_positives, false_negatives, precision, recall, f1_score, date_time)

    return precision, recall, f1_score


def heatmap(net, date_time, number_of_epoches):
    precision, recall, f1_score = loop_epoches(net, date_time, number_of_epoches)
    return precision, recall, f1_score