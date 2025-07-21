from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import scipy.ndimage
from skimage.transform import resize


class CustomDataset(Dataset):
    def __init__(self,data_dir,mode, train_min=None, train_max=None):
        self.data_dir = data_dir

        self.all_data = np.load(self.data_dir)
        if mode =='train':
            self.data = torch.tensor(self.all_data['train'], dtype=torch.float32)
            self.train_min = torch.min(self.data)
            self.train_max = torch.max(self.data)
            # print("train_min:",self.train_min)
            # print("train_max:",self.train_max)
        elif mode =='val':
            self.data = torch.tensor(self.all_data['val'], dtype=torch.float32)
            self.train_min = train_min
            self.train_max = train_max
        elif mode =='test':
            self.data = torch.tensor(self.all_data['test'], dtype=torch.float32)
            self.train_min = train_min
            self.train_max = train_max

    def __getitem__(self,i):
        x = self.data[i,:,:,:]
        x = 2*(x-self.train_min)/(self.train_max - self.train_min) -1
        return x

    def __len__(self):
        return self.data.shape[0]

    def get_min_max(self):
        """Return the minimum and maximum value in the dataset."""
        return torch.min(self.data), torch.max(self.data)