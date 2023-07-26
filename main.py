from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import nn as nn
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights


def train(model, device, train_loader, optimizer, epoch):
    model.train()  # for Dropout, Batch Normalization, etc.

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class PlantSeedlingDataset(Dataset):

    def __init__(self, data_root, transforms, stage):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}
        self.stage = stage

        data_root = Path(data_root) / stage
        for item in data_root.iterdir():
            if stage == 'train':
                if item.is_dir():
                    label = item.name
                    index = len(self.label_to_index)
                    self.label_to_index[label] = index
                    self.index_to_label[index] = label

                    for sub_item in item.iterdir():
                        if sub_item.is_file():
                            self.image_paths.append(sub_item)
                            self.labels.append(index)
            elif stage == 'test':
                if item.is_file():
                    self.image_paths.append(item)

        self.transforms = transforms

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]).convert('RGB') as image:
            if self.transforms:
                image = self.transforms(image)

        if self.stage == 'train':
            return image, self.labels[index]
        elif self.stage == 'test':
            return image

    def __len__(self):
        return len(self.image_paths)


def main():
    model = Model()
    device = torch.device('cuda')
    train_set = ...
    train_loader = ...
    train(model, device, train_loader, optimizer, epoch)


if __name__ == '__main__':
    main()


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet50(ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 12)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
