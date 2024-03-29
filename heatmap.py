import torch
import os
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
import plot


def find_file(path, filename):
    while path != '/' and path != '':
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate
        path = os.path.dirname(path)
    return None


def get_orig_position(liste):
    """Calculate original position of predicted coordinate."""
    liste[:, 0] = (np.floor(((liste[:, 0] + 0.5 + 1 + 1) * (883/442) + 1 + 1) * 2 + 2 - 0.5)) * (54/32)
    liste[:, 1] = (np.floor(((liste[:, 1] + 0.5 + 1 + 1) * (1179/590) + 1 + 1) * 2 + 2 - 0.5)) * (54/32)
    return liste


def create_score(img_name, epoch, list_label, list_max, num_pred, true_positives, false_positives, false_negatives):
    """Match predicted coordinates and marked coordinates. Calculate scores (recall, precision, F1 score)."""
    i = 0
    if (list_max.size == 0):
        prec = 0.0
        rec = 0.0
        f1 = 0.0
    else:
        while i < len(list_label):
            target = list_label[i]
            print(target)
            max_distance = 27
            distances = np.linalg.norm(list_max-target, axis=1)
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
        
        num_pred += len(list_max)
        false_positives = num_pred - true_positives
    
    return num_pred, true_positives, false_positives, false_negatives


def create_heatmaps(img_name, epoch, original_image, output_greater, image_max, mask, list_label, list_max, net_coord, date_time):
    """Create different heatmaps to visualize intermediate steps and result."""
    # output of network
    plot.plot_outputs(img_name, epoch, original_image, output_greater, date_time)
    
    # results of non-maximum suppression
    plot.plot_nonmax(img_name, epoch, original_image, image_max, date_time)

    # heatmap overlaid onto raw image
    plot.plot_transparent_heatmap_overlay(img_name, epoch, original_image, output_greater, list_label, list_max, date_time)

    # list_max and list_label coordinates overlaid onto heatmap
    plot.plot_label_max_coordinates_with_transparent_heatmap_overlay(img_name, epoch, original_image, output_greater, list_label, list_max, date_time)
    
    # compare points of list_label to points of list_max
    plot.plot_label_max_coordinates(img_name, epoch, original_image, list_label, list_max, date_time)


def create_coord_lists(number_of_epochs, json_filename, image_filename, img_name, net, epoch, num_pred, true_positives, false_positives, false_negatives, date_time):
    """Generate lists of predicted positions and ground truth positions."""
    im = skimage.io.imread(image_filename)
    if len(im.shape) == 3:
        im = rgb2gray(im)
    else:
        im = im /255.0
    im = im[:,:]

    original_image = im.copy()
    print('im.shape: {}\nim.dtype: {}\n im.max: {}'.format(im.shape, im.dtype, im.max()))
    im = skimage.transform.rescale(im, scale=(32 / 54))
    im = im.reshape((1, 1, im.shape[0], im.shape[1]))
    im = im.astype(np.float32)
    net.eval()
    print('im.shape: ', im.shape)
    with torch.no_grad():
        output = net(torch.tensor(im).cuda())
        predicted = output > 0.96
        print('output.shape: ', output.shape)
        print('output: {}\t predicted: {}'.format(output, predicted))

    output = output.data.cpu().numpy()
    output_greater = output[0, 0].copy()

    image_max = ndi.maximum_filter(output_greater.copy(), size=8, mode='constant')
    mask = output_greater == image_max
    mask &= output_greater > 0.96

    net_coord = np.nonzero(mask)
    net_coord = np.fliplr(np.column_stack(net_coord))

    list_max = get_orig_position(net_coord.copy())

    j = json.load(open(json_filename))
    red_pixels = j["red"]
    blue_pixels = j["blue"]
    
    list_label = red_pixels + blue_pixels
    list_label.sort()
    list_label = np.array(list_label)
    
    print('*** list_max (maxima of net_coord) ***\n{}\n*** list_label (marked bees) ***\n{}'.format(list_max, list_label))

    if (list_max.size != 0 and epoch == number_of_epochs-1):
        create_heatmaps(img_name, epoch, original_image, output_greater, image_max, mask, list_label, list_max, net_coord, date_time)

    num_pred, true_positives, false_positives, false_negatives = create_score(img_name, epoch, list_label, list_max, num_pred, true_positives, false_positives, false_negatives)

    return num_pred, true_positives, false_positives, false_negatives
    

def loop_epochs(given_arguments, net, date_time, number_of_epochs):
    """Loop over each epoch for every raw image in the original_test set."""
    f1_score = []
    precision = []
    recall = []

    save_epoch = 0

    for epoch in range(number_of_epochs):
        save_epoch = epoch
        print('*** Create heatmaps using state of epoch {} ***'.format(epoch))
        num_pred = 0.0
        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        for json_filename in given_arguments:
            j = json.load(open(json_filename))
            image_filename = find_file(os.path.dirname(json_filename), j["filename"])
            if image_filename is None:
                print("Missing image file for {}".format(json_filename))
                sys.exit(1)
            dirname, img_name = os.path.split(image_filename.replace('.png', ''))
            print(img_name)
        
            net.load_state_dict(torch.load('../Training/training_epoch-{}'.format(save_epoch)))
            num_pred, true_positives, false_positives, false_negatives = create_coord_lists(number_of_epochs, json_filename, image_filename, img_name, net, epoch, num_pred, true_positives, false_positives, false_negatives, date_time)
            
        if (false_positives == 0 and true_positives == 0):
            prec = 0
        else:
            prec = true_positives / (true_positives + false_positives)
        if (false_negatives == 0 and true_positives == 0):
            rec = 0
        else:
            rec = true_positives / (true_positives + false_negatives)
        numerator = prec * rec
        denominator = prec + rec
        if (denominator == 0):
            f1 = 0
        else:
            f1 = 2 * (numerator / denominator)
 
        precision += [prec]
        recall += [rec]
        f1_score += [f1]
        print('Precision:', prec)
        print('Recall:', rec)
        print('F1_Score:', f1)
        if epoch == (number_of_epochs-1):
            print('Test Epoch: %d\n#predicted: {}\n#true positives: {}\n#false positives: {}\n#false negatives: {}'.format(num_pred, true_positives, false_positives, false_negatives) % (epoch+1))

    print('F1_Score: ', f1_score)

    return precision, recall, f1_score


def heatmap(given_arguments, net, date_time, number_of_epochs):
    precision, recall, f1_score = loop_epochs(given_arguments, net, date_time, number_of_epochs)
    return precision, recall, f1_score