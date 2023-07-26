import torch

from main import Model


def test_model():
    inputs = torch.rand(2, 3, 224, 224).to('cuda')
    model = Model()
    model.to('cuda')
    outputs = model(inputs)
    assert outputs.shape == torch.Size([2, 12])
