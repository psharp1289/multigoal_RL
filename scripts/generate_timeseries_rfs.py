from random import sample
from numpy.random import choice
from random import random
from random import shuffle
import scipy.special as sf
import numpy as np
import pandas as pd
import scipy as sp
num_states=2
num_features=2
num_trials=80

reward_functions=np.zeros((num_trials,num_states))

#predominately safe
#types_rewards=[[[3,0,0],'safe'],[[0,0,-3],'predator'],[[0,0,-3],'predator'],[[3,0,0],'safe'],[[3,0,0],'safe'],[[0,1,0],'mild']]

#predominately dangerous
types_rewards=[[[0,-3],'predator'],[[0,-3],'predator'],[[0,-3],'predator'],[[3,0],'safe']]
types_rewards=[[[3,0],'safe'],[[3,0],'safe'],[[3,0],'safe'],[[0,-3],'predator']]


rew_over_time=[]
while rew_over_time.count('predator')!=28.0 and rew_over_time.count('safe')!=52.0:
    rew_over_time=[]
    for i in range(num_trials):
        current_reward_info=sample(types_rewards,1)[0]
        current_reward_function=current_reward_info[0]
        rew_over_time.append(current_reward_info[1])
        reward_functions[i]=current_reward_function

print('safe times:')
print(rew_over_time.count('safe'))
print('predator times:')
print(rew_over_time.count('predator'))

np.save('rf_safe_80trials.npy',reward_functions)

# reward_function_timeseries=np.load('rf_safe_thresh_short.npy')
# #build for logistic regression
# safe_rfs=[]
# mild_rfs=[]
# predator_rfs=[]
# for i in reward_function_timeseries:
#     if 3 in i:
#         safe_rfs.append(1)
#         mild_rfs.append(0)
#         predator_rfs.append(0)
#     elif -3 in i:
#         safe_rfs.append(0)
#         mild_rfs.append(0)
#         predator_rfs.append(1)
#     else:
#         safe_rfs.append(0)
#         mild_rfs.append(1)
#         predator_rfs.append(0)


#reward_function_timeseries=reward_functions 
# print(reward_function_timeseries)
# 
# print(reward_function_timeseries)
# stop
#np.save('rf_pred_thresh',reward_function_timeseries)

# # Run simulation
# In[9]: