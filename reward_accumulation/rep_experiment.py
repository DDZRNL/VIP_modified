import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from model_vip import VIP
from CustomEnv import CustomEnvWrapper

env_name = "HalfCheetah-v2"
device = "cuda"

# model loader
rep_model = VIP(device="cuda", lr=1e-4, hidden_dim=1024, size=50, l2weight=0.0, l1weight=0.0, gamma=0.98, num_negatives=0)
PATH="/data/htw/research/ours/evaluation/models/ours/halfcheetah/HC_M/snapshot_10000.pt"
vip_state_dict = torch.load(PATH, map_location=torch.device(device))['vip']
adjusted_state_dict = {k.replace('module.', ''): v for k, v in vip_state_dict.items()}
rep_model.load_state_dict(adjusted_state_dict)
rep_model.eval()

file_name = env_name + "_" + PATH.split('/')[-4]+"_"+PATH.split('/')[-2]+".pkl"

# build the env
env = gym.make(env_name)
# env = make_vec_env(env_name, n_envs=1)
wrapped_env = CustomEnvWrapper(env, rep_model)

model = PPO("MlpPolicy", wrapped_env, verbose=1)

env = model.get_env()

total_steps = 1_000_000     # 1_000_000
steps_per_train = 10_000    # 10_000 
times_per_eval = 10     # 10
eval_steps = 1000       # 1000


return_list = []
for _ in range(total_steps//steps_per_train):   #totally 1_000_000 steps
    model.learn(steps_per_train) # train 10_000 steps and then evaluate the policy
    # Train the policy 
    # Manually perform the training loop
    # n_steps = 2048
    # for i in range(steps_per_train):  # Assuming you want 10 training iterations
    #     # Collect data
    #     obs = env.reset()
    #     obs_tensor = torch.tensor(obs, dtype=torch.float)
    #     obs_rep = rep_model(obs_tensor)
    #     obs_rep = obs_rep.detach().numpy()
    #     for j in range(n_steps):
    #         action, _states = model.predict(obs_rep, deterministic=False)
    #         new_obs, rewards, dones, info = env.step(action)
    #         model.collect_rollouts(env, n_steps=n_steps, reset_num_timesteps=False)
            
    #         obs = new_obs
    #         obs_tensor = torch.tensor(obs, dtype=torch.float)
    #         obs_rep = rep_model(obs_tensor)
    #         obs_rep = obs_rep.detach().numpy()

    #     # Perform PPO optimization
    #     model.train()


    R = 0
    # run 10 times and average
    for i in range(times_per_eval):
        obs = env.reset()
        # obs_tensor = torch.tensor(obs[0], dtype=torch.float)
        # dim = len(obs_tensor)
        # obs_rep = rep_model(obs_tensor.view(1,dim))
        # obs_rep = obs_rep.detach().numpy()

        for j in range(eval_steps): # 1000 steps per episode
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            R += rewards
            # obs_tensor = torch.tensor(obs[0], dtype=torch.float)
            # dim = len(obs_tensor)
            # obs_rep = rep_model(obs_tensor.view(1,dim))
            # obs_rep = obs_rep.detach().numpy()
    R = R/times_per_eval
    return_list.append(R)


# save the return list
with open("rewards/"+file_name, 'wb') as f:
    pickle.dump(return_list, f)



