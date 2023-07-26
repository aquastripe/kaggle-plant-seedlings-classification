import torch

from main import PlantSeedlingDataset
from torchvision.transforms import transforms as T


def test_dataset():
    data_root = '/dataset/plant-seedlings-classification/'
    transforms = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])
    dataset = PlantSeedlingDataset(data_root, transforms)
    image, label = dataset

    assert image.shape == torch.Size([3, 224, 224])
    assert label.shape == torch.Size([1])
