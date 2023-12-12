import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

from .configs import data_configs, path_configs
from .dataset import ImageDataset


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def get_dataloader(dataset, train=True):
    data_cfg = data_configs()
    val_size = 0.1
    if train:
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_size)

        indices = list(range(dataset_size))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=2,
            # persistent_workers=True,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            sampler=val_sampler,
            drop_last=True,
            num_workers=2,
            # persistent_workers=True,
        )

        return train_loader, val_loader
    else:
        test_loader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            # num_workers=19,
            # persistent_workers=True,
        )
        return test_loader


def get_dataset():
    data_cfg = data_configs()
    path_cfg = path_configs()
    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(data_cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    test_size = 0.2
    dataset = ImageDataset(path_cfg.data_path, path_cfg.csv_path, transform)
    dataset_size = len(dataset)
    test_size = int(dataset_size * test_size)
    train_size = dataset_size - test_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return trainset, testset
