import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt
import pickle

env_name = "HalfCheetah-v2"

env = gym.make(env_name, render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1)

env = model.get_env()

total_steps = 1_000_000
steps_per_train = 10_000
times_per_eval = 10
eval_steps = 1000

obs = env.reset()
return_list = []
for _ in range(total_steps//steps_per_train):   #totally 1_000_000 steps
    model.learn(steps_per_train)   # train 10_000 steps and then evaluate the policy
    R = 0
    # run 10 times and average
    for i in range(times_per_eval):
        obs = env.reset()
        for j in range(eval_steps): # 1000 steps per episode
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            R += rewards
    R = R/times_per_eval
    return_list.append(R)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(' {}'.format(env_name))
plt.savefig("test.jpg")
plt.show()

# save the return list
with open('reward.pkl', 'wb') as f:
    pickle.dump(return_list, f)



