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

import d4rl
import gym
from utils.utils import return_range, torchify

def get_env_and_dataset(env_name, max_episode_steps=1000):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d', 'ant')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = v #torchify(v)

    return dataset

## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, task, datatype="expert", num_workers=10, doaug = "none"):
        self._num_workers = max(1, num_workers)
        self.task = task
        self.datatype = datatype
        self.env_name = task + "-" + datatype + "-v2"
        assert(task is not None)
        self.doaug = doaug
        
        # Load Data
        self.dataset = get_env_and_dataset(self.env_name)

    def _sample(self):

        total_length = len(self.dataset['observations'])

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, total_length-2)
        end_ind = np.random.randint(start_ind+1, total_length)

        s0_ind = np.random.randint(start_ind, end_ind)
        # s1_ind_vip = min(s0_ind_vip+1, end_ind)
        
        reward = self.dataset["rewards"][s0_ind]

        ### Encode each image individually
        if(self.task == 'ant'):
            im0 = self.dataset['observations'][start_ind][:27]
            img = self.dataset['observations'][end_ind][:27]
            imts0 = self.dataset['observations'][s0_ind][:27]
            imts1 = self.dataset['next_observations'][s0_ind][:27]
        else:
            im0 = self.dataset['observations'][start_ind]
            img = self.dataset['observations'][end_ind]
            imts0 = self.dataset['observations'][s0_ind]
            imts1 = self.dataset['next_observations'][s0_ind]

        terminal = self.dataset["terminals"][s0_ind]

        im = torch.stack([torch.tensor(im0), torch.tensor(img), torch.tensor(imts0), torch.tensor(imts1)])
        #im = self.preprocess(im)
        return (im, reward, terminal)

    def __iter__(self):
        while True:
            yield self._sample()
