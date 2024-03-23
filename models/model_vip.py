# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

class VIP(nn.Module):
    def __init__(self, device="cuda", lr=1e-3, hidden_dim=1024, size=50, l2weight=1.0, l1weight=1.0, gamma=0.98, num_negatives=0, input_dim=17, output_dim=4):
        super().__init__()
        self.device = device
        self.l2weight = l2weight
        self.l1weight = l1weight

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.size = size # Resnet size
        self.num_negatives = num_negatives

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        # hyperparameter
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), # HC 17; ant 27
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # 256
        )

        self.mlp.train()
        params += list(self.mlp.parameters())

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr = lr)

    ## Forward Call (im --> representation)
    def forward(self, obs):
        obs_p = F.normalize(obs, dim = 1)
        h = self.mlp(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        d = -torch.linalg.norm(tensor1 - tensor2, dim = -1)
        return d
    
