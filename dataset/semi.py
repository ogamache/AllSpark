from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from util.utils import init_log
import cv2

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.probability_transform = 0.2
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=self.probability_transform),
                A.HorizontalFlip(p=self.probability_transform),
                A.VerticalFlip(p=self.probability_transform),
                A.RandomBrightnessContrast(p=self.probability_transform),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
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
        if self.name == "potsdam":
            id = self.ids[item]
            img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

            if self.mode == 'val':
                img, mask = normalize(img, mask)
                return img, mask, id

            img, mask = resize(img, mask, (0.5, 2.0))
            ignore_value = 254 if self.mode == 'train_u' else 255
            img, mask = crop(img, mask, self.size, ignore_value)
            img, mask = hflip(img, mask, p=0.5)

            if self.mode == 'train_u':
                return normalize(img)

            img_tf = deepcopy(img)
            mask_tf = deepcopy(mask)

            # if random.random() < 0.8:
            #     img_tf = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_tf)
            # img_tf = transforms.RandomGrayscale(p=0.2)(img_tf)
            # img_tf = blur(img_tf, p=0.5)
            # img_tf, mask = normalize(img_tf, mask_tf)

            transformed = self.transform(image=np.array(img_tf), mask=np.array(mask_tf))
            img_tf = transformed["image"].long()
            mask_tf = transformed["mask"].long()

            return img_tf, mask_tf
        else:
            logger = init_log('semi')
            logger.propagate = 0
            id = self.ids[item]
            img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            masks = self.__fetch_all_masks(self.root, id.split(' ')[1])

            if self.mode == 'val':
                img = normalize(img)
                ## Function to merge all mask into one deep array
                mask = torch.from_numpy(np.array(mask)).long()
                return img, mask, id

            img, masks = resize(img, masks, (0.5, 2.0))
            logger.info(f"{masks}")
            ignore_value = 254 if self.mode == 'train_u' else 255
            img, masks = crop(img, masks, self.size, ignore_value)
            logger.info(f"{masks}")
            img, masks = hflip(img, masks, p=0.5)
            logger.info(f"{masks}")

            if self.mode == 'train_u':
                return normalize(img)

            img_tf = deepcopy(img)
            mask_tf = [np.array(m) for m in masks]
            logger.info(f"{mask_tf}")

            transformed = self.transform(image=np.array(img_tf), masks=mask_tf)
            img_tf = transformed["image"].long()
            mask_tf = transformed["masks"]
            logger.info(f"{mask_tf[0]}")

            ## function to merge all the masks into one deep image

            # # Visualize augmentation
            # tensor_to_pil = transforms.ToPILImage()
            # save_img_tf = tensor_to_pil(img_tf)
            # save_mask_tf = tensor_to_pil(mask_tf[0])
            # img.save("img_og.png")
            # masks[0].save("mask_og.png")
            # save_img_tf.save("img_tf.png")
            # save_mask_tf.save("mask_tf.png")

            return img_tf, mask_tf


    def __len__(self):
        return len(self.ids)
    
    def __fetch_all_masks(self, path):
        mask1 = Image.fromarray(np.array(Image.open(os.path.join(path))))
        mask2 = Image.fromarray(np.array(Image.open(os.path.join(path))))
        mask3 = Image.fromarray(np.array(Image.open(os.path.join(path))))
        return [mask1, mask2, mask3]
    
    def __combined_all_masks(self, masks):
        return
