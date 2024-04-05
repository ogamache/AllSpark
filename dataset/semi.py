from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        n_classes = len(id.split(' '))
        mask_arrays = []
        for i in range(n_classes):
            mask_arrays.append(np.array(Image.open(os.path.join(self.root, id.split(' ')[i+1]))))
        mask_tensor = np.stack(mask_arrays)

        # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        # mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask_tensor = normalize(img, mask_tensor)
            return img, mask_tensor, id

        img, mask_tensor = resize(img, mask_tensor, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask_tensor = crop(img, mask_tensor, self.size, ignore_value)
        img, mask_tensor = hflip(img, mask_tensor, p=0.5)

        if self.mode == 'train_u':
            return normalize(img)

        img_s1 = deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        img_s1, mask_tensor = normalize(img_s1, mask_tensor)

        # if self.mode == 'val':
        #     img, mask = normalize(img, mask)
        #     return img, mask, id

        # img, mask = resize(img, mask, (0.5, 2.0))
        # ignore_value = 254 if self.mode == 'train_u' else 255
        # img, mask = crop(img, mask, self.size, ignore_value)
        # img, mask = hflip(img, mask, p=0.5)

        # if self.mode == 'train_u':
        #     return normalize(img)

        # img_s1 = deepcopy(img)

        # if random.random() < 0.8:
        #     img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        # img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        # img_s1 = blur(img_s1, p=0.5)
        # img_s1, mask = normalize(img_s1, mask)
        return img_s1, mask_tensor

    def __len__(self):
        return len(self.ids)