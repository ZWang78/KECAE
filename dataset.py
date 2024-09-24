import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import numbers

def get_pair(I):
    s = I.size[0]
    pad = int(np.floor(s / 3))
    ps = 128
    l = I.crop([0, pad, ps, pad + ps])
    m = I.crop([s - ps, pad, s, pad + ps])
    m = m.transpose(Image.FLIP_LEFT_RIGHT)
    return l, m

class KneeGradingDataset(Dataset):
    def __init__(self, dataset, transform, augment, stage='train'):
        super(KneeGradingDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.augment = augment
        self.stage = stage
        if self.stage == 'train':
            self.images, self.labels = self.load_csv("paper_0_2_100000.csv")
        if self.stage == 'valid':
            self.images, self.labels = self.load_csv("trainData_0_2.csv")

    def load_csv(self, filename):
        images, labels = [], []
        with open(os.path.join(self.dataset, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_0, l_0, img_2, l_2 = row
                label_0 = int(l_0)
                label_2 = int(l_2)
                images.append([img_0, img_2])
                labels.append([label_0,label_2])
        return images, labels

    def __getitem__(self, index):
        img_0_path, img_2_path = self.images[index]
        label_0, label_2 = self.labels[index]

        # Construct full paths, assuming a directory structure like 'dataset/0/image.png'
        fname = os.path.join(self.dataset, img_0_path[0], img_0_path)
        fname2 = os.path.join(self.dataset, img_2_path[0], img_2_path)

        img = Image.open(fname)
        img2 = Image.open(fname2)

        img = self.augment(img)
        img2 = self.augment(img2)

        l, m = get_pair(img)
        l2, m2 = get_pair(img2)

        l = self.transform(l)
        m = self.transform(m)
        l2 = self.transform(l2)
        m2 = self.transform(m2)

        img = self.transform(img)
        img2 = self.transform(img2)

        return img, img2, l, m, l2, m2, label_0, label_2

    def __len__(self):
        return len(self.images)

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        tw, th, = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))