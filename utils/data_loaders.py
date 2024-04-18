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
        
        # self.flat_dataset = self.data_paths.flatten()
        
    def _sample(self):

        # Sample a video from datasource
        num_vid = len(self.data_paths)
        
        video_id = np.random.randint(0, int(num_vid))
        vid = self.data_paths[video_id]

        vidlen = len(vid)

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen-2)
        # end_ind = np.random.randint(start_ind+1, vidlen)

        s0_ind = np.random.randint(start_ind, vidlen-1)
        s1_ind = min(s0_ind+1, vidlen-1)
        
        ### Encode each image individually
        im0 = vid[start_ind]
        # img = vid[end_ind]
        imts0 = vid[s0_ind]
        imts1 = vid[s1_ind]

        prob = np.random.uniform(0, 1)
        if(prob <= 0.2):    # 20% choose goal as next_state
            end_ind = s1_ind
            img = vid[end_ind]
        elif(0.2 < prob and prob <=0.7):    # 50% choose goal as random state after curruent state in the same traj
            end_ind = np.random.randint(s0_ind+1, vidlen)
            img = vid[end_ind]
        else:   # 30% choose random goal
            # img = np.random.choice(self.flat_dataset)
            sublist = random.choice(self.data_paths)
            img = random.choice(sublist)

        # encode as torch
        im0 = torch.tensor(im0)
        img = torch.tensor(img)
        imts0 = torch.tensor(imts0)
        imts1 = torch.tensor(imts1)

        # Self-supervised reward (this is always -1)
        if(torch.equal(imts0, img)):
            reward = 0
        else:
            reward = -1
        # reward = float(s0_ind == end_ind) - 1

        im = torch.stack([im0, img, imts0, imts1])

        if(s1_ind == vidlen-1):
            terminal = 1.0
        else:
            terminal = 0.0

        return (im, reward, terminal)

    def __iter__(self):
        while True:
            yield self._sample()
