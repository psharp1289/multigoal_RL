import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr as corr
os.chdir('feature_randomwalks')
bandits=np.load('b1a.npy')
bandits2=np.load('b2a.npy')
bandits3=np.load('b3a.npy')
bandits4=np.load('b4a.npy')

bandits=np.concatenate((bandits,bandits2),axis=2)
bandits2=np.concatenate((bandits3,bandits4),axis=2)
all_bandits1=np.concatenate((bandits,bandits2),axis=2)
# all_bandits1=np.load('allbanditsused.npy')
bandits=all_bandits1
# print(corr(bandits[1,1,:],bandits[1,0,:]))
# print(corr(bandits[1,1,:],bandits[0,0,:]))
# print(corr(bandits[1,1,:],bandits[0,1,:]))
# print(corr(bandits[0,0,:],bandits[1,0,:]))
# print(corr(bandits[0,0,:],bandits[0,1,:]))
# print(corr(bandits[1,0,:],bandits[0,1,:]))

# all_bandits1=np.load('danger_firsta.npy')
# np.save('order1_2statesN_Ca',bandits)
# np.save('order2_2statesN_Ca',bandits2)
# np.save('safe_firsta',all_bandits)
# np.save('danger_firsta',all_bandits)

num_states=2
num_trials=160



# for state in range(num_states):
  # tbandits=all_bandits1[state]
tbandits=all_bandits1
timepoint=list(np.arange(1,num_trials+1,1))*4
all_bandits=[]
event=[]
feature_names=['reward1','punish1','reward2','punish2']
s=0
for j in range(num_states):
  for i in range(num_states):
     all_bandits+=list(tbandits[j][i])
     name_bandit='{}'.format(feature_names[s])
     s+=1
     event+=[name_bandit]*(num_trials)
print(len(all_bandits))
print(len(event))
fig = plt.figure()
colors=['gold','black','goldenrod','gray','gold','black','goldenrod','gray']
ax = sns.lineplot(x=timepoint, y=all_bandits, hue=event)
for i,j in enumerate(ax.lines):
  print(i)
  j.set_color(colors[i])
leg = ax.get_legend()
leg.legendHandles[0].set_color('gold')
leg.legendHandles[1].set_color('black')
leg.legendHandles[2].set_color('goldenrod')
leg.legendHandles[3].set_color('gray')
ax.set_xlabel("trial number",fontsize=12)
ax.set_ylabel("p(feature|action)",fontsize=12)
plt.savefig("feature_trajectories_all.png",bbox_inches='tight', dpi=300)
plt.show()

# for state in range(num_states):
#   tbandits=bandits2[state]
#   timepoint=list(np.arange(1,num_trials+1,1))*num_states
#   all_bandits=[]
#   event=[]
#   for i in range(num_states):
#      all_bandits+=list(tbandits[i])
#      name_bandit='feature {}'.format(i+1)
#      event+=[name_bandit]*(num_trials)
#   fig = plt.figure()

#   ax = sns.lineplot(x=timepoint, y=all_bandits, hue=event)
#   ax.set_title('State {}'.format(state+1))
