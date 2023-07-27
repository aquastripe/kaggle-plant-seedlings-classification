import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


def train(model, train_loader, valid_loader, optimizer, num_epochs, device):
    loss_records = {
        'train': [],
        'valid': [],
    }
    for epoch in tqdm(range(num_epochs), desc='Training Kaggle plant seedling dataset', ascii=True):
        with torch.enable_grad():
            model.train()  # for Dropout, Batch Normalization, etc.
            for inputs, targets in tqdm(train_loader, ascii=True, desc='Training'):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # forward
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_records['train'].append(loss.item())

        with torch.no_grad():
            # validation
            model.eval()
            for inputs, targets in tqdm(valid_loader, ascii=True, desc='Validating'):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # forward
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

                loss_records['valid'].append(loss.item())

        torch.save(model.state_dict(), f'model-{epoch}.pth')

    with open('loss.json', 'w', encoding='utf-8') as f:
        json.dump(loss_records, f)


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


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet50(ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 12)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--num_epochs', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda')
    model = Model()
    model.to(device)
    transforms = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])
    dataset = PlantSeedlingDataset(args.data_root, transforms, 'train')
    train_set, valid_set = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, train_loader, valid_loader, optimizer, args.num_epochs, device)


if __name__ == '__main__':
    main()
