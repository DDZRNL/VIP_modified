import torch
import torch.nn as nn
from .util import mlp


class TwinQ(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim * 2, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, next_state):
        sa = torch.cat([state, next_state], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, next_state):
        return torch.min(*self.both(state, next_state))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)