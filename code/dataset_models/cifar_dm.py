import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from .datasets import *

from global_config import *

class DataModuleCIFAR(pl.LightningDataModule):
    def __init__(self, download_dir, ratio, batch_size):
        super().__init__()
          
        self.download_dir = download_dir
        self.batch_size = batch_size
        self.ratio = ratio
        self.dataset_code = dataset_code

    self.transform = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.RandomCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

      
    def prepare_data(self):
        datasets.CIFAR10(self.download_dir, train=True, download=True)
        datasets.CIFAR10(self.download_dir, train=False, download=True)

  
    def setup(self, stage=None):
        self.train_set = CIFARDataset(
            self.download_dir,
            self.dataset_code,
            ratio=self.ratio,
            train=True, 
            transform=self.transform,
            download=False
        )
        
        self.valid_set = CIFARDataset(
            self.download_dir,
            self.dataset_code,
            ratio=self.ratio,
            train=False,
            transform=self.transform,
            download=False
        )

        print("train set size", len(self.train_set))
        print("valid set size", len(self.valid_set))
  
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=GPU_COUNT
        )
  
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=GPU_COUNT
        )

    def test_dataloader(self):
        self.test_set = self.valid_set

        return self.val_dataloader()
