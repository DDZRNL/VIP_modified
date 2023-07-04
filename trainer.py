# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import torchvision.transforms as T

from IQL.iql import ImplicitQLearning
from IQL.policy import GaussianPolicy, DeterministicPolicy
from IQL.value_functions import TwinQ, ValueFunction
from IQL.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilon = 1e-8
def do_nothing(x): return x

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

class Trainer():
    def __init__(self, eval_freq, obs_dim, optimizer_factory):
        self.eval_freq = eval_freq
        self.tau = 0.7
        self.discount=0.99
        self.alpha=0.005
        self.vf = ValueFunction(obs_dim, hidden_dim=64, n_hidden=2).to(DEFAULT_DEVICE)
        self.qf = TwinQ(obs_dim, hidden_dim=64, n_hidden=2).to(DEFAULT_DEVICE)
        self.v_target = copy.deepcopy(self.vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())  
        self.q_optimizer = optimizer_factory(self.qf.parameters())

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        b_im, b_reward = batch
        t2 = time.time()

        ## Encode Start and End Frames
        bs = b_im.shape[0]
        img_stack_size = b_im.shape[1]
        H = b_im.shape[-2]
        W = b_im.shape[-1]
        b_im_r = b_im.reshape(bs*img_stack_size, 3, H, W)
        alles = model(b_im_r)
        alle = alles.reshape(bs, img_stack_size, -1)
        e0 = alle[:, 0] # initial, o_0
        eg = alle[:, 1] # final, o_g
        es0_vip = alle[:, 2] # o_t
        es1_vip = alle[:, 3] # o_t+1

        # e0_target = copy.deepcopy(e0)
        # eg_target = copy.deepcopy(eg)
        # es0_vip_target = copy.deepcopy(es0_vip)
        # es1_vip_target = copy.deepcopy(es1_vip)

        full_loss = 0

        # obs_dim = e0.shape[1]
        # print(e0.shape) # [32, 2]

        with torch.no_grad():
            target_v = self.v_target(es0_vip, eg)
            # next_v = self.vf(es1_vip)
        
        ## LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        full_loss += model.module.l2weight * l2loss
        full_loss += model.module.l1weight * l1loss
        t3 = time.time()

        ## VIP Loss 
        V_0 = model.module.sim(e0, eg) # -||phi(s) - phi(g)||_2
        terminal = model.module.sim(eg, eg)
        r =  b_reward.to(V_0.device) # R(s;g) = (s==g) - 1 
        # V_s = model.module.sim(es0_vip, eg)
        # V_s_next = model.module.sim(es1_vip, eg)
        V_s = self.vf(es0_vip, eg)
        V_s_next = self.vf(es1_vip, eg)
        
        # Update Q function     # function (6)
        targets = r + (1. - terminal.float()) * self.discount * V_s_next.detach() # target V_s_next
        qs = self.qf.both(es0_vip, es1_vip, eg)
        vip_loss = (V_s.mean() - asymmetric_l2_loss(self.qf(es0_vip, es1_vip, eg), self.tau))**2

        v_loss = torch.mean(self.qf(es0_vip, es1_vip, eg) - r - self.discount * V_s) ** 2
        v_loss = Variable(v_loss, requires_grad=True)

        q_loss = sum(F.mse_loss(q.float(), targets.float()) for q in qs) / len(qs)
        q_loss = Variable(q_loss, requires_grad=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward(retain_graph=True)   # retain_graph=True 
        self.q_optimizer.step()
        
        # update V
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward(retain_graph=True)
        self.v_optimizer.step()

        # update target V
        update_exponential_moving_average(self.v_target, self.vf, self.alpha)
        
        metrics['vip_loss'] = vip_loss.item()
        full_loss += vip_loss.detach()
        # full_loss += q_loss
        metrics['full_loss'] = full_loss.item()
        t4 = time.time()

        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()
        t5 = time.time()

        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP time {t3-t2}, VIP time {t4-t3}, Backprop time {t5-t4}"
        return metrics, st
