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

import torchqmet
from IQL.util import mlp

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilon = 1e-8

def do_nothing(x): return x

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

class Trainer():
    def __init__(self, model_target, eval_freq, obs_dim, optimizer_factory, v_hidden_dim, q_hidden_dim):
        self.eval_freq = eval_freq
        self.tau = 0.7
        self.discount=0.99
        self.alpha=0.005
        self.qf = TwinQ(obs_dim, hidden_dim=q_hidden_dim, n_hidden=2).to(DEFAULT_DEVICE)
        # self.vf = ValueFunction(obs_dim, hidden_dim=v_hidden_dim, n_hidden=2).to(DEFAULT_DEVICE) #QuasimetricModel(obs_dim, v_hidden_dim, 2, quasimetric_head_spec).to(DEFAULT_DEVICE)
        # self.vf_target = copy.deepcopy(self.vf).to(DEFAULT_DEVICE)
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        # self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.target_model = model_target

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
        alles = model(b_im)
        
        e0 = alles[:, 0] # initial, o_0
        eg = alles[:, 1] # final, o_g
        es0_vip = alles[:, 2] # o_t
        es1_vip = alles[:, 3] # o_t+1

        # e0_target = copy.deepcopy(e0)
        alles_target = self.target_model(b_im)
        eg_target = alles_target[:, 1] # final, o_g
        es0_vip_target = alles_target[:, 2] # o_t
        es1_vip_target = alles_target[:, 3] # o_t+1

        full_loss = 0

        # with torch.no_grad():
        V_s_target = self.target_model.sim(es0_vip_target, eg_target)
        V_s_next_target = self.target_model.sim(es1_vip_target, eg_target)

        # LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        full_loss += model.l2weight * l2loss
        full_loss += model.l1weight * l1loss
        t3 = time.time()

        ## VIP Loss 
        V_0 = model.sim(e0, eg)  # -||phi(s) - phi(g)||_2
        r =  b_reward.to(V_0.device)    # R(s;g) = (s==g) - 1  
        V_s = model.sim(es0_vip, eg)
        V_s_next = model.sim(es1_vip, eg)

        if(torch.equal(es0_vip, eg)):
            terminal = 1.0
        else:
            terminal = 0.0
        
        ### Adv = Q(s,s’,g).detach - V(s,g) 
        ### V_loss = Expectile loss(adv, tu)   
        Adv = self.qf(es0_vip, es1_vip, eg).detach() - V_s
        vip_loss = asymmetric_l2_loss(Adv, self.tau)
        
        metrics['vip_loss'] = vip_loss.item()
        full_loss += vip_loss
        metrics['full_loss'] = full_loss.item()
        t4 = time.time()

        if not eval:
            model.encoder_opt.zero_grad()
            full_loss.backward()
            model.encoder_opt.step()
            # update target V
            update_exponential_moving_average(self.target_model, model, self.alpha)
        t5 = time.time()

        # Update Q function     # function (6)
        # targets = r + (1. - terminal) * self.discount * V_s_next.detach()
        qs = self.qf(es0_vip, es1_vip, eg) # self.qf.both(es0_vip, es1_vip, eg)
        ### Q_loss = Q(s,s’,g) - （r + γV_target(s’,g).detach()） 
        # q_loss = MSE(qs -(r + (1. - terminal) * self.discount * V_s_next_target.detach())) # torch.mean
        q_loss = F.mse_loss(qs, (r + (1. - terminal) * self.discount * V_s_next_target.detach()))
        q_loss = Variable(q_loss, requires_grad=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward(retain_graph=True)   # retain_graph=True 
        self.q_optimizer.step()
        metrics['q_loss'] = q_loss.item()

        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP time {t3-t2}, VIP time {t4-t3}, Backprop time {t5-t4}"
        return metrics, st
