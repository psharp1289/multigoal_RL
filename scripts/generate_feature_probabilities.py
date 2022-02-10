def generate_bandits(starting_bandits,num_trials,num_momentum_switches,random_walk,momentum_mean,momentum_variance):
  import numpy as np
  from random import shuffle
  num_bandits=len(starting_bandits)
  shuffle(starting_bandits)
  value_scale=1

  #define momentum pace
  momentum_switches=np.arange(0,num_trials,round(num_trials/num_momentum_switches))
  bandits=[]
  #generate random walks for each bandit
  for bandit in range(num_bandits):
    exec('global bandit_{};bandit_{}=[]'.format(bandit,bandit))
    exec('bandit_{}.append(starting_bandits[bandit])'.format(bandit))
    current_random_walks=[]
    momentum_changes=[]
    momentum_counter=0
    for trial in range(num_trials):
      current_random_walks.append(np.random.normal(0,random_walk))
      exec('new_bandit=[bandit_{}[trial]+current_random_walks[{}]]'.format(bandit,trial))
      exec('new_bandit=[0.2 if x<0.2 else x for x in new_bandit]')
      exec('new_bandit=[0.8 if x>0.8 else x for x in new_bandit]')
      exec('new_bandit=new_bandit[0]')
      exec('bandit_{}.append(new_bandit)'.format(bandit))
    exec('bandits.append(bandit_{})'.format(bandit))

  return bandits

def generate_bandits2(starting_bandits,num_trials,num_momentum_switches,random_walk,momentum_mean,momentum_variance):
  import numpy as np
  from random import shuffle
  num_bandits=len(starting_bandits)
  value_scale=1

  #define momentum pace
  momentum_switches=np.arange(0,num_trials,round(num_trials/num_momentum_switches))
  bandits=[]
  #generate random walks for each bandit
  for bandit in range(3):
    exec('global bandit_{};bandit_{}=[]'.format(bandit,bandit))
    exec('bandit_{}.append(starting_bandits[bandit])'.format(bandit))
    current_random_walks=[]
    momentum_changes=[]
    momentum_counter=0
    for trial in range(num_trials):
      current_random_walks.append(np.random.normal(0,random_walk))
      exec('new_bandit=[bandit_{}[trial]+current_random_walks[{}]]'.format(bandit,trial))
      exec('new_bandit=[0.2 if x<0.2 else x for x in new_bandit]')
      exec('new_bandit=[0.8 if x>0.8 else x for x in new_bandit]')
      exec('new_bandit=new_bandit[0]')
      exec('bandit_{}.append(new_bandit)'.format(bandit))
    exec('bandits.append(bandit_{})'.format(bandit))

  return bandits



#generate bandits
# # Generate feature random walks

# In[10]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy.stats import spearmanr as corrs
from random import shuffle
import numpy as np
num_trials=140
num_states=3
diff_probs=1
max_corr=1
max_corr2=1
diff_probstwo=1
iteration=1
for i in range(2):
  if i==0:
    print('start block 1 while loop')
    while max_corr>0.5 or diff_probs>0.05:
      corrs_est=[]
      iteration+=1
      bandits=np.zeros((3,3,num_trials))
      for state in range(num_states):
        bandit_vals=[0.4,0.5,0.6]
        shuffle(bandit_vals)
        num_bandits=len(bandit_vals)
        momentum_switches=1
        momentum_mean=0
        momentum_variance=0.00001
        random_walk_variance=0.03

        tbandits=generate_bandits(bandit_vals,num_trials-1,momentum_switches,random_walk_variance,momentum_mean,momentum_variance)
        #print(len(tbandits[0]))
        bandits[state]=tbandits
      safe_avg_prob=np.mean(bandits[0][0])+np.mean(bandits[1][0])+np.mean(bandits[2][0])
      pred_avg_prob=np.mean(bandits[0][2])+np.mean(bandits[1][2])+np.mean(bandits[2][2])
      med_avg_prob=np.mean(bandits[0][1])+np.mean(bandits[1][1])+np.mean(bandits[2][1])

      diff_probs1=np.abs(safe_avg_prob-pred_avg_prob)
      diff_probs2=np.abs(safe_avg_prob-med_avg_prob)
      diff_probs3=np.abs(med_avg_prob-pred_avg_prob)
      diff_probs=np.max([diff_probs1,diff_probs2,diff_probs3])

      #compute and save relevant correlations
      #correlation within features
      r1,p1=corrs(bandits[0][0],bandits[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][0],bandits[2][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][0],bandits[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][2],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][2],bandits[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][2],bandits[1][2])
      corrs_est.append(np.abs(r1))

      #correlation between features

      #safe and pred
      r1,p1=corrs(bandits[0][0],bandits[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][0],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][0],bandits[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][0],bandits[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][0],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][0],bandits[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][0],bandits[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][0],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][0],bandits[2][2])


      #safe and mild
      r1,p1=corrs(bandits[0][1],bandits[0][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][1],bandits[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][1],bandits[2][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][1],bandits[0][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][1],bandits[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][1],bandits[2][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][1],bandits[0][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][1],bandits[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][1],bandits[2][0])

      #pred and mild
      r1,p1=corrs(bandits[0][1],bandits[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][1],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[0][1],bandits[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][1],bandits[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][1],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[1][1],bandits[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][1],bandits[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][1],bandits[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits[2][1],bandits[2][2])

      corrs_est.append(np.abs(r1))
      max_corr=np.max(corrs_est)
  if i==1:
    print('start block 2 while loop')
    while max_corr2>0.5 or diff_probstwo>0.05:
      corrs_est=[]
      iteration+=1
      bandits2=np.zeros((3,3,num_trials))
      for state in range(num_states):
        bandit_vals=[bandits[state][0][num_trials-1] ,bandits[state][1][num_trials-1],bandits[state][2][num_trials-1]]
        num_bandits2=len(bandit_vals)
        momentum_switches=1
        momentum_mean=0
        momentum_variance=0.00001
        random_walk_variance=0.03
        tbandits2=generate_bandits2(bandit_vals,num_trials-1,momentum_switches,random_walk_variance,momentum_mean,momentum_variance)
        #print(len(tbandits2[0]))
        bandits2[state]=tbandits2
      safe_avg_prob=np.mean(bandits2[0][0])+np.mean(bandits2[1][0])+np.mean(bandits2[2][0])
      pred_avg_prob=np.mean(bandits2[0][2])+np.mean(bandits2[1][2])+np.mean(bandits2[2][2])
      med_avg_prob=np.mean(bandits2[0][1])+np.mean(bandits2[1][1])+np.mean(bandits2[2][1])

      diff_probs1=np.abs(safe_avg_prob-pred_avg_prob)
      diff_probs2=np.abs(safe_avg_prob-med_avg_prob)
      diff_probs3=np.abs(med_avg_prob-pred_avg_prob)
      diff_probstwo=np.max([diff_probs1,diff_probs2,diff_probs3])

      #compute and save relevant correlations
      #correlation within features
      r1,p1=corrs(bandits2[0][0],bandits2[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][0],bandits2[2][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][0],bandits2[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][2],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][2],bandits2[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][2],bandits2[1][2])
      corrs_est.append(np.abs(r1))

      #correlation between features

      #safe and pred
      r1,p1=corrs(bandits2[0][0],bandits2[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][0],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][0],bandits2[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][0],bandits2[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][0],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][0],bandits2[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][0],bandits2[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][0],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][0],bandits2[2][2])


      #safe and mild
      r1,p1=corrs(bandits2[0][1],bandits2[0][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][1],bandits2[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][1],bandits2[2][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][1],bandits2[0][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][1],bandits2[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][1],bandits2[2][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][1],bandits2[0][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][1],bandits2[1][0])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][1],bandits2[2][0])

      #pred and mild
      r1,p1=corrs(bandits2[0][1],bandits2[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][1],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[0][1],bandits2[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][1],bandits2[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][1],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[1][1],bandits2[2][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][1],bandits2[0][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][1],bandits2[1][2])
      corrs_est.append(np.abs(r1))
      r1,p1=corrs(bandits2[2][1],bandits2[2][2])

      corrs_est.append(np.abs(r1))
      max_corr2=np.max(corrs_est)
#plot bandit trajectories
print('final max corr r1: {}'.format(max_corr))
print('final max corr 2: {}'.format(max_corr2))
# import numpy as np
# # num_trials=200
# bandits=np.load('dangerous_first.npy')
# bandits2=np.load('high_reward_second.npy')
# bandits=np.dstack((bandits,bandits2))
# import matplotlib.pyplot as plt
# import seaborn as sns
# num_bandits=3

# for state in range(num_bandits):
#   tbandits=bandits[state]
#   timepoint=list(np.arange(1,num_trials+1,1))*num_bandits
#   all_bandits=[]
#   event=[]
#   for i in range(num_bandits):
#      all_bandits+=list(tbandits[i])
#      name_bandit='feature {}'.format(i+1)
#      event+=[name_bandit]*(num_trials)
#   fig = plt.figure()

#   ax = sns.lineplot(x=timepoint, y=all_bandits, hue=event)
#   ax.set_title('State {}'.format(state+1))
#   plt.show()

# for state in range(num_bandits):
#   tbandits=bandits2[state]
#   timepoint=list(np.arange(1,num_trials+1,1))*num_bandits
#   all_bandits=[]
#   event=[]
#   for i in range(num_bandits):
#      all_bandits+=list(tbandits[i])
#      name_bandit='feature {}'.format(i+1)
#      event+=[name_bandit]*(num_trials)
#   fig = plt.figure()

#   ax = sns.lineplot(x=timepoint, y=all_bandits, hue=event)
#   ax.set_title('State {}'.format(state+1))
#   plt.show()

np.save('order1',bandits)
np.save('order2',bandits2)
# np.save('feature_probabilities_block_strict_lowMI_b2',bandits2)