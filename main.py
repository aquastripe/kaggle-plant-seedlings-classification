import argparse
import pandas as pd
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
    data_loader = {
        'train': train_loader,
        'valid': valid_loader,
    }
    desc = {
        'train': 'Training',
        'valid': 'Validating',
    }
    for epoch in tqdm(range(num_epochs), desc='Training Kaggle plant seedling dataset', ascii=True):
        for stage in ['train', 'valid']:
            in_training = stage == 'train'
            with torch.set_grad_enabled(in_training):
                model.train(in_training)  # for Dropout, Batch Normalization, etc.
                for inputs, targets in tqdm(data_loader[stage], ascii=True, desc=desc[stage]):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    # forward
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)

                    # backward
                    if in_training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    loss_records[stage].append(loss.item())

        torch.save(model.state_dict(), f'model-{epoch}.pth')

    with open('loss.json', 'w', encoding='utf-8') as f:
        json.dump(loss_records, f)


class PlantSeedlingDataset(Dataset):

    def __init__(self, data_root, transforms, stage):
        super().__init__()
        self.image_paths = []
        self.labels = []

        metadata_file = 'metadata.json'
        if stage == 'train':
            self.label_to_index = {}
            self.index_to_label = {}
        elif stage == 'test':
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.label_to_index = metadata['label_to_index']
                self.index_to_label = metadata['index_to_label']

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

        if stage == 'train':
            with open(metadata_file, 'w', encoding='utf-8') as f:
                metadata = {
                    'label_to_index': self.label_to_index,
                    'index_to_label': self.index_to_label,
                }
                json.dump(metadata, f)

        self.transforms = transforms

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]).convert('RGB') as image:
            if self.transforms:
                image = self.transforms(image)

        if self.stage == 'train':
            return image, self.labels[index]
        elif self.stage == 'test':
            return image, self.image_paths[index].name

    def __len__(self):
        return len(self.image_paths)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 12)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def evaluate(model, test_loader, device):
    files = []
    species = []
    model.eval()
    for inputs, filenames in tqdm(test_loader, desc='Testing', ascii=True):
        inputs = inputs.to(device, non_blocking=True)

        with torch.no_grad():
            # forward
            outputs = model(inputs)

            # evaluation, e.g.
            label_indices = outputs.argmax(dim=1)
            files += filenames
            species += [test_loader.dataset.index_to_label[index] for index in label_indices]

    return files, species


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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, train_loader, valid_loader, optimizer, args.num_epochs, device)

    test_set = PlantSeedlingDataset(args.data_root, transforms, 'test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    files, species = evaluate(model, test_loader, device)
    df = pd.DataFrame({'file': files, 'species': species})
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
