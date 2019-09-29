import numpy as np
import scipy.stats
import torch.utils.data as data
import torch

class MyDataset(data.Dataset):
    def __init__(self, dataset, sigma, transform):
        self.sigma = sigma
        self.probability_distribution = scipy.stats.norm(scale=self.sigma)
        self.probability_at_zero = self.probability_distribution.pdf(0.0)
        self.dataset = dataset
        self.transformer = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        total_offset = 0.0
        offset = [0.0, 0.0]

        if label == 0:
            one_hot = torch.tensor((0))
            label_strength=0
        else:
            offset = np.random.normal(0, scale=self.sigma, size=(2,))
            while (offset[0] > 15 or offset[1] > 15 or offset[0] < -15 or offset[1] < -15):
                offset = np.random.normal(0, scale=self.sigma, size=(2,))
            cx = 39 + offset[0]
            cy = 39 + offset[1]
            image = image.crop((cx - 39, cy - 39, cx + 39, cy + 39))

            offset_norm = np.linalg.norm(offset)
            label_strength = self.probability_distribution.pdf(offset_norm) / self.probability_at_zero
            one_hot = torch.tensor((label_strength))

        image = self.transformer(image)

        return image, label_strength