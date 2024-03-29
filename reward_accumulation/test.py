import gymnasium as gym

from stable_baselines3 import PPO

# Parallel environments
env_name = "HalfCheetah-v2"

env = gym.make(env_name)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10)


obs = env.reset()
print(obs)
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(obs)