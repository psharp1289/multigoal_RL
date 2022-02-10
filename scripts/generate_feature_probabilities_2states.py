def generate_bandits(starting_bandits,num_trials,num_momentum_switches,random_walk,momentum_mean,momentum_variance):
  import numpy as np
  from random import shuffle
  from random import sample
  num_states=len(starting_bandits)
  value_scale=1

  #define momentum pace
  momentum_switches=np.arange(0,num_trials,round(num_trials/num_momentum_switches))
  bandits=[]
  #generate random walks for each bandit
  for bandit in range(num_states):
    exec('global bandit_{};bandit_{}=[]'.format(bandit,bandit))
    exec('bandit_{}.append(starting_bandits[bandit])'.format(bandit))
    current_random_walks=[]
    momentum_changes=[]
    momentum_counter=0
    for trial in range(num_trials):
      # if trial==40:
      #       current_random_walks.append(sample([-0.3,0.3],1)[0])
      # else:
      current_random_walks.append(np.random.normal(0,random_walk))
      
      exec('new_bandit=[bandit_{}[trial]+current_random_walks[{}]]'.format(bandit,trial))
      exec('new_bandit=[0.25 if x<0.25 else x for x in new_bandit]')
      exec('new_bandit=[0.75 if x>0.75 else x for x in new_bandit]')
      exec('new_bandit=new_bandit[0]')
      exec('bandit_{}.append(new_bandit)'.format(bandit))
    exec('bandits.append(bandit_{})'.format(bandit))

  return bandits

def generate_bandits2(starting_bandits,num_trials,num_momentum_switches,random_walk,momentum_mean,momentum_variance):
  import numpy as np
  from random import sample 
  from random import shuffle
  num_states=len(starting_bandits)
  value_scale=1

  #define momentum pace
  momentum_switches=np.arange(0,num_trials,round(num_trials/num_momentum_switches))
  bandits=[]
  #generate random walks for each bandit
  for bandit in range(2):
    exec('global bandit_{};bandit_{}=[]'.format(bandit,bandit))
    exec('bandit_{}.append(starting_bandits[bandit])'.format(bandit))
    current_random_walks=[]
    momentum_changes=[]
    momentum_counter=0
    for trial in range(num_trials):
      # if trial==40:
      #       current_random_walks.append(sample([-0.3,0.3],1)[0])
      # else:
      current_random_walks.append(np.random.normal(0,random_walk))
      
      exec('new_bandit=[bandit_{}[trial]+current_random_walks[{}]]'.format(bandit,trial))
      exec('new_bandit=[0.25 if x<0.25 else x for x in new_bandit]')
      exec('new_bandit=[0.75 if x>0.75 else x for x in new_bandit]')
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
num_trials=40
avg_diff=0
avg_diff2=0
avg_diff_second=0
avg_diff_second1=0
avg_diff_second2=0
avg_diff_third=0
avg_diff_third1=0
avg_diff_third2=0
avg_diff_fourth=0
avg_diff_fourth1=0
avg_diff_fourth2=0
avg_cross1=0
avg_cross2=0
avg_cross3=0
avg_cross4=0
end_diff=100
end_diff2=100
num_states=2
num_features=2
diff_probs=1
avg_diff_blocks1=0
avg_diff_blocks2=0
max_corr=1
max_corr2=1
max_corr3=1
max_corr4=1
diff_probstwo=1
iteration=1
#
print('start block 1 while loop')
while max_corr>0.6 or diff_probs>0.10 or avg_diff>0.05 or avg_diff1<0.27 or avg_diff2<0.27 or avg_diff1>0.35 or avg_diff2>0.35 or end_diff>0.10 or end_diff2>0.10:
  corrs_est=[]
  iteration+=1
  bandits=np.zeros((num_states,num_features,num_trials))
  for state in range(num_states):
    if state==0:
      bandit_vals=[0.60,0.60]
    else:
      bandit_vals=[0.40,0.40]
    num_states=len(bandit_vals)
    momentum_switches=1
    momentum_mean=0
    momentum_variance=0.00001
    random_walk_variance=0.04

    tbandits=generate_bandits(bandit_vals,num_trials-1,momentum_switches,random_walk_variance,momentum_mean,momentum_variance)
    #print(len(tbandits[0]))
    bandits[state]=tbandits
  safe_avg_prob=np.mean(bandits[0][0])+np.mean(bandits[1][0]) #+np.mean(bandits[2][0])
  pred_avg_prob=np.mean(bandits[0][1])+np.mean(bandits[1][1]) #+np.mean(bandits[2][2])

  diff_probs1=np.abs(safe_avg_prob-pred_avg_prob)
  diff_probs=diff_probs1

  avg_diff1=np.abs(np.mean(bandits[0][0])-np.mean(bandits[1][0]))
  avg_diff2=np.abs(np.mean(bandits[0][1])-np.mean(bandits[1][1]))
  end_diff=np.abs(np.mean(bandits[0][0][39])-np.mean(bandits[0][1][39]))
  end_diff2=np.abs(np.mean(bandits[1][0][39])-np.mean(bandits[1][1][39]))
  avg_diff=np.abs(avg_diff1-avg_diff2)
  # diff_probs2=np.abs(safe_avg_prob-med_avg_prob)
  # diff_probs3=np.abs(med_avg_prob-pred_avg_prob)
  # diff_probs=np.max([diff_probs1,diff_probs2,diff_probs3])

  #compute and save relevant correlations
  #correlation within features
  r1,p1=corrs(bandits[0][0],bandits[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits[0][1],bandits[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits[0][1],bandits[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits[1][1],bandits[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits[1][1],bandits[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits[1][1],bandits[0][1])
  corrs_est.append(np.abs(r1))

  max_corr=np.max(corrs_est)



   
print('start block 2 while loop')
while max_corr2>0.6 or diff_probstwo>0.05 or avg_diff_second>0.05 or avg_diff_second1<0.27 or avg_diff_second1>0.35 or avg_diff_second2<0.27 or avg_diff_second2>0.35 or avg_cross1<0.10 or end_diff2>0.10 or end_diff>0.10:
  corrs_est=[]
  iteration+=1

  bandits2=np.zeros((num_states,num_features,num_trials))
  for state in range(num_states):
    bandit_vals=[bandits[np.abs(state)][0][num_trials-1] ,bandits[np.abs(state)][1][num_trials-1]]
    num_states2=len(bandit_vals)
    momentum_switches=1
    momentum_mean=0
    momentum_variance=0.00001
    random_walk_variance=0.04
    tbandits2=generate_bandits2(bandit_vals,num_trials-1,momentum_switches,random_walk_variance,momentum_mean,momentum_variance)
    #print(len(tbandits2[0]))
    bandits2[state]=tbandits2
  safe_avg_prob=np.mean(bandits2[0][0])+np.mean(bandits2[1][0]) #+np.mean(bandits2[2][0])
  pred_avg_prob=np.mean(bandits2[0][1])+np.mean(bandits2[1][1]) #+np.mean(bandits2[2][2])
  # med_avg_prob=np.mean(bandits2[0][1])+np.mean(bandits2[1][1])+np.mean(bandits2[2][1])

  diff_probs1=np.abs(safe_avg_prob-pred_avg_prob)
  diff_probstwo=diff_probs1

  avg_diff_second1=np.abs(np.mean(bandits2[0][0])-np.mean(bandits2[1][0]))
  avg_diff_second2=np.abs(np.mean(bandits2[0][1])-np.mean(bandits2[1][1]))
  avg_diff_second=np.abs(avg_diff_second1-avg_diff_second2)
  end_diff=np.abs(np.mean(bandits2[0][0][39])-np.mean(bandits2[0][1][39]))
  end_diff2=np.abs(np.mean(bandits2[1][0][39])-np.mean(bandits2[1][1][39]))

  avg_cross1=np.abs(np.mean(bandits[0][0])-np.mean(bandits2[0][0]))
  avg_cross2=np.abs(np.mean(bandits[0][1])-np.mean(bandits2[0][1]))
  avg_cross3=np.abs(np.mean(bandits[1][0])-np.mean(bandits2[1][0]))
  avg_cross4=np.abs(np.mean(bandits[1][1])-np.mean(bandits2[1][1]))
  # diff_probs2=np.abs(safe_avg_prob-med_avg_prob)
  # diff_probs3=np.abs(med_avg_prob-pred_avg_prob)
  # diff_probstwo=np.max([diff_probs1,diff_probs2,diff_probs3])

  #compute and save relevant correlations
  #correlation within features
  r1,p1=corrs(bandits2[0][1],bandits2[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits2[0][1],bandits2[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits2[0][0],bandits2[1][0])
  corrs_est.append(np.abs(r1))
  # r1,p1=corrs(bandits2[0][0],bandits2[0][1])
  # corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits2[1][1],bandits2[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits2[1][1],bandits2[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits2[1][1],bandits2[0][1])
  corrs_est.append(np.abs(r1))



   
  max_corr2=np.max(corrs_est)
print('start block 3 while loop')
while max_corr3>0.6 or diff_probstwo>0.05 or avg_diff_third>0.05 or avg_diff_third1<0.27 or avg_diff_third1>0.35 or avg_diff_third2<0.27 or avg_diff_third2>0.35 or avg_cross1<0.10 or end_diff2>0.10 or end_diff>0.10:
  corrs_est=[]
  iteration+=1

  bandits3=np.zeros((num_states,num_features,num_trials))
  for state in range(num_states):
    bandit_vals=[bandits2[np.abs(state)][0][num_trials-1] ,bandits2[np.abs(state)][1][num_trials-1]]
    num_states2=len(bandit_vals)
    momentum_switches=1
    momentum_mean=0
    momentum_variance=0.00001
    random_walk_variance=0.04
    tbandits3=generate_bandits2(bandit_vals,num_trials-1,momentum_switches,random_walk_variance,momentum_mean,momentum_variance)
    #print(len(tbandits3[0]))
    bandits3[state]=tbandits3
  safe_avg_prob=np.mean(bandits3[0][0])+np.mean(bandits3[1][0]) #+np.mean(bandits3[2][0])
  pred_avg_prob=np.mean(bandits3[0][1])+np.mean(bandits3[1][1]) #+np.mean(bandits3[2][2])
  # med_avg_prob=np.mean(bandits3[0][1])+np.mean(bandits3[1][1])+np.mean(bandits3[2][1])
  end_diff=np.abs(np.mean(bandits3[0][0][39])-np.mean(bandits3[0][1][39]))
  end_diff2=np.abs(np.mean(bandits3[0][1][39])-np.mean(bandits3[1][1][39]))

  diff_probs1=np.abs(safe_avg_prob-pred_avg_prob)
  diff_probstwo=diff_probs1

  avg_diff_third1=np.abs(np.mean(bandits3[0][0])-np.mean(bandits3[1][0]))
  avg_diff_third2=np.abs(np.mean(bandits3[0][1])-np.mean(bandits3[1][1]))
  avg_diff_third=np.abs(avg_diff_third1-avg_diff_third2)

  avg_cross1=np.abs(np.mean(bandits2[0][0])-np.mean(bandits3[0][0]))
  avg_cross2=np.abs(np.mean(bandits2[0][1])-np.mean(bandits3[0][1]))
  avg_cross3=np.abs(np.mean(bandits2[1][0])-np.mean(bandits3[1][0]))
  avg_cross4=np.abs(np.mean(bandits2[1][1])-np.mean(bandits3[1][1]))
  # diff_probs2=np.abs(safe_avg_prob-med_avg_prob)
  # diff_probs3=np.abs(med_avg_prob-pred_avg_prob)
  # diff_probstwo=np.max([diff_probs1,diff_probs2,diff_probs3])

  #compute and save relevant correlations
  #correlation within features
  r1,p1=corrs(bandits3[0][1],bandits3[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits3[0][1],bandits3[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits3[0][0],bandits3[1][0])
  corrs_est.append(np.abs(r1))
  # r1,p1=corrs(bandits3[0][0],bandits3[0][1])
  # corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits3[1][1],bandits3[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits3[1][1],bandits3[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits3[1][1],bandits3[0][1])
  corrs_est.append(np.abs(r1))
   
  max_corr3=np.max(corrs_est)

print('start block 4 while loop')
while max_corr4>0.6 or diff_probstwo>0.05 or avg_diff_fourth>0.05 or avg_diff_fourth1<0.27 or avg_diff_fourth1>0.35 or avg_diff_fourth2<0.27 or avg_diff_fourth2>0.35 or avg_cross1<0.10:
  corrs_est=[]
  iteration+=1

  bandits4=np.zeros((num_states,num_features,num_trials))
  for state in range(num_states):
    bandit_vals=[bandits3[np.abs(state)][0][num_trials-1] ,bandits3[np.abs(state)][1][num_trials-1]]
    num_states2=len(bandit_vals)
    momentum_switches=1
    momentum_mean=0
    momentum_variance=0.00001
    random_walk_variance=0.04
    tbandits4=generate_bandits2(bandit_vals,num_trials-1,momentum_switches,random_walk_variance,momentum_mean,momentum_variance)
    #print(len(tbandits4[0]))
    bandits4[state]=tbandits4
  safe_avg_prob=np.mean(bandits4[0][0])+np.mean(bandits4[1][0]) #+np.mean(bandits4[2][0])
  pred_avg_prob=np.mean(bandits4[0][1])+np.mean(bandits4[1][1]) #+np.mean(bandits4[2][2])
  # med_avg_prob=np.mean(bandits4[0][1])+np.mean(bandits4[1][1])+np.mean(bandits4[2][1])

  diff_probs1=np.abs(safe_avg_prob-pred_avg_prob)
  diff_probstwo=diff_probs1

  avg_diff_fourth1=np.abs(np.mean(bandits4[0][0])-np.mean(bandits4[1][0]))
  avg_diff_fourth2=np.abs(np.mean(bandits4[0][1])-np.mean(bandits4[1][1]))
  avg_diff_fourth=np.abs(avg_diff_fourth1-avg_diff_fourth2)

  avg_cross1=np.abs(np.mean(bandits3[0][0])-np.mean(bandits4[0][0]))
  avg_cross2=np.abs(np.mean(bandits3[0][1])-np.mean(bandits4[0][1]))
  avg_cross3=np.abs(np.mean(bandits3[1][0])-np.mean(bandits4[1][0]))
  avg_cross4=np.abs(np.mean(bandits3[1][1])-np.mean(bandits4[1][1]))
  # diff_probs2=np.abs(safe_avg_prob-med_avg_prob)
  # diff_probs3=np.abs(med_avg_prob-pred_avg_prob)
  # diff_probstwo=np.max([diff_probs1,diff_probs2,diff_probs3])

  #compute and save relevant correlations
  #correlation within features
  r1,p1=corrs(bandits4[0][1],bandits4[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits4[0][1],bandits4[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits4[0][0],bandits4[1][0])
  corrs_est.append(np.abs(r1))
  # r1,p1=corrs(bandits4[0][0],bandits4[0][1])
  # corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits4[1][1],bandits4[0][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits4[1][1],bandits4[1][0])
  corrs_est.append(np.abs(r1))
  r1,p1=corrs(bandits4[1][1],bandits4[0][1])
  corrs_est.append(np.abs(r1))



   
  max_corr4=np.max(corrs_est)
#plot bandit trajectories

print('final max corr r1: {}'.format(max_corr))
print('final diff between differences in trajectories r1: {}'.format(avg_diff))
print('')
print('final max corr 2: {}'.format(max_corr2))
print('final diff between differences in trajectories r2: {}'.format(avg_diff_second))
print('')
print('final max corr 3: {}'.format(max_corr3))
print('final diff between differences in trajectories r2: {}'.format(avg_diff_second))
print('')
print('final max corr 4: {}'.format(max_corr4))
print('final diff between differences in trajectories r2: {}'.format(avg_diff_second))
print('')
# np.save('b1b',bandits)
# np.save('b2b',bandits2)
# np.save('b3b',bandits3)
# np.save('b4b',bandits4)

bandits=np.concatenate((bandits,bandits2),axis=2)
bandits2=np.concatenate((bandits3,bandits4),axis=2)
all_bandits=np.concatenate((bandits,bandits2),axis=2)
# np.save('order1_2statesN_C',bandits)
# np.save('order2_2statesN_C',bandits2)
# np.save('safe_first',all_bandits)
# np.save('danger_first',all_bandits)

import numpy as np
# num_trials=200
# bandits=np.load('dangerous_first.npy')
# bandits2=np.load('high_reward_second.npy')
# bandits=np.dstack((bandits,bandits2))
import matplotlib.pyplot as plt
import seaborn as sns
# num_states=3

for state in range(num_states):
  tbandits=bandits[state]
  timepoint=list(np.arange(1,num_trials+1,1))*num_states
  all_bandits=[]
  event=[]
  for i in range(num_states):
     all_bandits+=list(tbandits[i])
     name_bandit='feature {}'.format(i+1)
     event+=[name_bandit]*(num_trials)
  fig = plt.figure()

  ax = sns.lineplot(x=timepoint, y=all_bandits, hue=event)
  ax.set_title('State {}'.format(state+1))
  plt.show()

for state in range(num_states):
  tbandits=bandits2[state]
  timepoint=list(np.arange(1,num_trials+1,1))*num_states
  all_bandits=[]
  event=[]
  for i in range(num_states):
     all_bandits+=list(tbandits[i])
     name_bandit='feature {}'.format(i+1)
     event+=[name_bandit]*(num_trials)
  fig = plt.figure()

  ax = sns.lineplot(x=timepoint, y=all_bandits, hue=event)
  ax.set_title('State {}'.format(state+1))
  plt.show()

# all_bandits=np.concatenate((bandits,bandits2),axis=2)
# np.save('order1_2statesN_C',bandits)
# np.save('order2_2statesN_C',bandits2)
# np.save('safe_first',all_bandits)
# np.save('danger_first',all_bandits)

# np.save('feature_probabilities_block_strict_lowMI_b2',bandits2)