import pytest
import torch
from torch.utils.data import DataLoader

from main import PlantSeedlingDataset
from torchvision.transforms import transforms as T


def test_train_set():
    data_root = '/dataset/plant-seedlings-classification/'
    transforms = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])
    dataset = PlantSeedlingDataset(data_root, transforms, stage='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    images, labels = next(iter(dataloader))

    assert images.shape == torch.Size([2, 3, 224, 224])
    assert labels.shape == torch.Size([2])


def test_test_set():
    data_root = '/dataset/plant-seedlings-classification/'
    transforms = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])
    dataset = PlantSeedlingDataset(data_root, transforms, stage='test')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    images = next(iter(dataloader))

    assert images.shape == torch.Size([2, 3, 224, 224])


@pytest.mark.skip(reason='Not implemented')
def test_dataset_size():
    data_root = '/dataset/plant-seedlings-classification/'
    transforms = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])
    dataset = PlantSeedlingDataset(data_root, transforms)
    assert len(dataset) == ...
