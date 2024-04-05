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

from IQL.util import mlp

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilon = 1e-8

def do_nothing(x): return x

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

# moving average.  momentum update
def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

class Trainer():
    def __init__(self, model_target, eval_freq, obs_dim, optimizer_factory, v_hidden_dim, q_hidden_dim):
        self.eval_freq = eval_freq
        self.tau = 0.9
        self.discount=0.99
        self.alpha=0.005
        self.qf = TwinQ(obs_dim, hidden_dim=q_hidden_dim, n_hidden=2).to(DEFAULT_DEVICE)
        self.qf_target = copy.deepcopy(self.qf).to(DEFAULT_DEVICE)
        self.vf = ValueFunction(obs_dim, hidden_dim=v_hidden_dim, n_hidden=2).to(DEFAULT_DEVICE) #QuasimetricModel(obs_dim, v_hidden_dim, 2, quasimetric_head_spec).to(DEFAULT_DEVICE)
        # self.vf_target = copy.deepcopy(self.vf).to(DEFAULT_DEVICE)
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.encoder_q_target = model_target

    def update(self, model, encoder_q, batch, step, eval=False):
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
        alles_target = self.encoder_q_target(b_im)
        eg_target = alles_target[:, 1] # final, o_g
        es0_vip_target = alles_target[:, 2] # o_t
        es1_vip_target = alles_target[:, 3] # o_t+1
        
        alles_q = encoder_q(b_im)
        eg_q = alles_q[:, 1] # final, o_g
        es0_vip_q = alles_q[:, 2] # o_t
        es1_vip_q = alles_q[:, 3] # o_t+1

        full_loss = 0

        # with torch.no_grad():
        # V_s_target = self.target_model.sim(es0_vip_target, eg_target)
        # V_s_next_target = self.target_model.sim(es1_vip_target, eg_target)

        # LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        full_loss += model.l2weight * l2loss
        full_loss += model.l1weight * l1loss
        t3 = time.time()

        ## V Loss 
        V_0 = self.vf(e0, eg)  # -||phi(s) - phi(g)||_2
        r =  b_reward.to(V_0.device)    # R(s;g) = (s==g) - 1  
        V_s = self.vf(es0_vip, eg)
        V_s_next = self.vf(es1_vip, eg)

        if(torch.equal(es0_vip, eg)):
            terminal = 1.0
        else:
            terminal = 0.0
        
        ### Adv = Q(s,s’,g).detach - V(s,g) 
        ### V_loss = Expectile loss(adv, tu)   
        Adv = self.qf_target(es0_vip_target, es1_vip_target, eg_target).detach() - V_s
        V_loss = asymmetric_l2_loss(Adv, self.tau)
        
        metrics['V_loss'] = V_loss.item()
        full_loss += V_loss
        metrics['full_loss'] = full_loss.item()
        t4 = time.time()

        if not eval:
            model.encoder_opt.zero_grad()
            self.v_optimizer.zero_grad()
            full_loss.backward()
            model.encoder_opt.step()
            self.v_optimizer.step()
        t5 = time.time()

        # Update Q function   
        targets = r + (1. - terminal) * self.discount * V_s_next.detach()
        qs = self.qf(es0_vip_q, es1_vip_q, eg_q) # self.qf.both(es0_vip, es1_vip, eg)
        ### Q_loss = Q(s,s’,g) - （r + γV_target(s’,g).detach()） 
        # q_loss = MSE(qs -(r + (1. - terminal) * self.discount * V_s_next_target.detach())) # torch.mean
        # q_loss = F.mse_loss(qs, (r + (1. - terminal) * self.discount * V_s_next.detach()))
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        q_loss = Variable(q_loss, requires_grad=True)
        if not eval:
            encoder_q.encoder_opt.zero_grad()   # set_to_none=True
            self.q_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)   # retain_graph=True
            encoder_q.encoder_opt.step()
            self.q_optimizer.step()
            update_exponential_moving_average(self.encoder_q_target, encoder_q, self.alpha)
            update_exponential_moving_average(self.qf_target, self.qf, self.alpha)
        
        metrics['q_loss'] = q_loss.item()

        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP time {t3-t2}, VIP time {t4-t3}, Backprop time {t5-t4}"
        return metrics, st
