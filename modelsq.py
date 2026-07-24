"""Real-valued fully connected networks executed on the photonic twin.

Intensity-encoded inputs (nonnegative), signed weights on differential
rails; ReLU between layers keeps activations nonnegative for intensity
re-encoding at the next layer, with a linear output layer.
"""
import torch
import torch.nn as nn
from physicsq import PhotonChain


class RealFC(nn.Module):
    def __init__(self, dims=(784, 300, 100, 10)):
        super().__init__()
        self.dims = dims
        self.weights = nn.ParameterList()
        for n, m in zip(dims[:-1], dims[1:]):
            w = torch.randn(m, n) * (2.0 / n) ** 0.5
            self.weights.append(nn.Parameter(w))

    @property
    def layer_dims(self):
        return list(zip(self.dims[:-1], self.dims[1:]))

    def forward(self, x, chain: PhotonChain = None):
        if chain is None:
            chain = PhotonChain()
        L = len(self.weights)
        for i, W in enumerate(self.weights):
            x = chain.mvm(x, W)
            if i < L - 1:
                x = torch.relu(x)
        return x
