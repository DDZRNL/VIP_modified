import matplotlib.pyplot as plt
import pickle

env_name = "HalfCheetah-v2"
df = open("reward.pkl", 'rb')
return_list = pickle.load(df)
df.close()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(' {}'.format(env_name))
plt.savefig("graph.jpg")
plt.show()

