from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.probability_transform = 0.5
        self.transform = A.Compose(
            [
                A.ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
                A.ToGray(p=0.2),
                A.Blur(p=self.probability_transform),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True, p=1.0),
                ToTensorV2(transpose_mask=True),
            ], p=1.0
        )
        self.transform_val_and_unlabeled = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True, p=1.0),
                ToTensorV2(transpose_mask=True),
            ], p=1.0
        )

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
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            transformed_val = self.transform_val_and_unlabeled(image=np.array(img), mask=np.array(mask))
            img = transformed_val["image"]
            mask = transformed_val["mask"].long()
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_u':
            transformed_unlabeled = self.transform_val_and_unlabeled(image=np.array(img))
            img = transformed_unlabeled["image"]
            return img

        img_tf = deepcopy(img)
        mask_tf = deepcopy(mask)
        

        transformed = self.transform(image=np.array(img_tf), mask=np.array(mask_tf))
        img_after_data_aug = transformed["image"]
        mask_after_data_aug = transformed["mask"].long()

        return img_after_data_aug, mask_after_data_aug

    def __len__(self):
        return len(self.ids)
