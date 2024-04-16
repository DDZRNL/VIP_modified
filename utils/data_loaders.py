# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, datasource='ego4d', datapath=None, num_workers=10, doaug = "none"):
        self._num_workers = max(1, num_workers)
        self.datasource = datasource
        self.datapath = datapath
        assert(datapath is not None)
        self.doaug = doaug
        # self.data_paths = np.load(self.datapath, allow_pickle=True)
        with open(self.datapath, 'rb') as f:
            self.data_paths = pickle.load(f)
        
        # Load Data

    def _sample(self):

        # Sample a video from datasource
        num_vid = len(self.data_paths)

        video_id = np.random.randint(0, int(num_vid))
        vid = self.data_paths[video_id]

        vidlen = len(vid)

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen-2)
        end_ind = np.random.randint(start_ind+1, vidlen)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip+1, end_ind)
        
        # Self-supervised reward (this is always -1)
        reward = float(s0_ind_vip == end_ind) - 1

        ### Encode each image individually
        im0 = vid[start_ind]
        img = vid[end_ind]
        imts0_vip = vid[s0_ind_vip]
        imts1_vip = vid[s1_ind_vip]

        # print("The data loader:")
        im = torch.stack([torch.tensor(im0), torch.tensor(img), torch.tensor(imts0_vip), torch.tensor(imts1_vip)])
        return (im, reward)

    def __iter__(self):
        while True:
            yield self._sample()
