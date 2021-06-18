import h5py
import torch
import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

from PIL import Image

from global_config import *


class CIFARDataset(Dataset):
	""" Dataset for all MNIST-like datasets """
	def __init__(self, root, dataset_code, train=True, transform=None, target_transform=None, download=False):
		super().__init__()

		dataset = CIFAR10(root=root, train=train, download=download, transform=transform, target_transform=target_transform)

		self.transform = transform
		self.target_transform = target_transform
		self.ratio = ratio
		self.dataset_code = dataset_code

		self.data = dataset.data
		self.targets = dataset.targets if dataset_code != SVHN_CODE else dataset.labels

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):
		img, target = self.data[index], int(self.targets[index])

		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target
