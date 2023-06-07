#%%
import pandas as pd
import matplotlib.pyplot as plt
   
data1 = pd.read_csv('../data/qtrain.csv')
data2 = pd.read_csv('../data/qeval.csv')
data3 = pd.read_csv('../data/strain.csv')
data4 = pd.read_csv('../data/seval.csv')
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)

#plt.plot(df1['step'], df1['episode_reward'], color='red', marker='o', label='Qlearning')
plt.plot(df2['step'], df2['episode_reward'], color='red', marker='o', label='Qlearning')
#plt.plot(df3['step'], df3['episode_reward'], color='blue', marker='o', label='SARSA')
plt.plot(df4['step'], df4['episode_reward'], color='blue', marker='o', label='SARSA')
#plt.title('Agent performance Qlearning (train)', fontsize=14)
#plt.title('Agent performance Qlearning (eval)', fontsize=14)
#plt.title('Agent performance SARSA (train)', fontsize=14)
#plt.title('Agent performance SARSA (eval)', fontsize=14)
#plt.title('Agent performance Qlearning vs SARSA (train)', fontsize=14)
plt.title('Agent performance Qlearning vs SARSA (eval)', fontsize=14)
plt.xlabel('step', fontsize=14)
plt.ylabel('Episode Reward', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

#%%