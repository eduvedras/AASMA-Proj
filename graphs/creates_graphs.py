#%%
import pandas as pd
import matplotlib.pyplot as plt
   
#data = pd.read_csv('../experiment/2023.05.27/1959_bc_orl-vanilla/eval.csv')
data = pd.read_csv('../experiment/2023.05.29/1608_qlearning_vanilla/eval.csv')
df = pd.DataFrame(data)

plt.plot(df['step'], df['episode_reward'], color='red', marker='o')
plt.title('Agent performance Qlearning (train)', fontsize=14)
plt.xlabel('step', fontsize=14)
plt.ylabel('Episode Reward', fontsize=14)
plt.grid(True)
plt.show()

#%%