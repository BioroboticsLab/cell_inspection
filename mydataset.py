import numpy as np
import scipy.stats
import torch.utils.data as data
import torch

class MyDataset(data.Dataset):
    """Translate bee details from the training set by a random offset."""
    def __init__(self, dataset, sigma, transform):
        # sigma determines shape of labelstrength over offset length
        self.sigma = sigma
        self.probability_distribution = scipy.stats.norm(scale=self.sigma)
        self.probability_at_zero = self.probability_distribution.pdf(0.0)
        self.dataset = dataset
        self.transformer = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        offset_norm = 0.0
        offset = [0.0, 0.0]
        label_strength = 0.0

        if label == 1:
            # scale determines shape of offset distribution (see offset graph)
            offset = np.random.normal(0, scale=4, size=(2,))
            while (offset[0] > 15 or offset[1] > 15 or offset[0] < -15 or offset[1] < -15):
                offset = np.random.normal(0, scale=4, size=(2,))
            # the positive training details are 130x130px images
            # their center point is [65,65]
            cx = 65 + offset[0]
            cy = 65 + offset[1]
            image = image.crop((cx - 39, cy - 39, cx + 39, cy + 39))

            offset_norm = np.linalg.norm(offset)
            label_strength = self.probability_distribution.pdf(offset_norm) / self.probability_at_zero
            
        label = torch.tensor((label_strength, offset_norm, offset[0], offset[1]))
        image = self.transformer(image)

        return image, label
