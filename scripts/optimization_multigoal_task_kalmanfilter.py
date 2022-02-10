'''


'''
import concurrent.futures
import time
from itertools import repeat
from multiprocessing import Pool
import multiprocessing as mp
import scipy.special as sf
import numpy as np
from numpy.random import gamma
from numpy.random import beta
from numpy.random import uniform
from numpy.random import normal
from scipy.stats import pearsonr as r
import scipy as sp
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from feature_learning_functions import simulate_one_step_featurelearner_beta_parallel
from feature_learning_functions import simulate_one_step_featurelearner_counterfactual_parallel_2states
from feature_learning_functions import simulate_2iRF_sticky
from feature_learning_functions import simulate_2iRF_sticky_flipped
from feature_learning_functions import simulate_2iRF_sticky_pflipped
from feature_learning_functions import simulate_2iRF_sticky_rflipped
from feature_learning_functions import simulate_2iRF_sticky_punishments
from feature_learning_functions import simulate_2iRF_sticky_rewards
from feature_learning_functions import simulate_VKF_binary
'''
generate simulated data

'''
import itertools
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Generate values for each parameter
# inputs: num_subjects,num_trials,lrf,lrv,beta1,beta2,mf_betas,lr_cfs,sticks,pred_changes,rew_changes,rfi_rews,rfi_preds,bandits,rew_func,games_played


num_subjects=[1]
num_trials=[160]
omega1= np.arange(0.01,0.3,0.005)
# omega2= np.arange(0,10,2)
# omega3= np.arange(0,10,2)
# omega4= np.arange(0,10,2)



# beta3=np.arange(0,6,1)
bandits=np.load('safe_firsta.npy')

bandits=[bandits]

rf2=np.load('rf_safe_80trials.npy')
rf2=rf2/3.0
rf1=np.load('rf_pred_80trials.npy')
rf1=rf1/3.0

rew_func=np.concatenate((rf2,rf1),axis=0)
# rew_func=np.concatenate((rf2,rf1),axis=0)
# rew_func=rew_func*-1
	



rew_func=[rew_func]
games=500
games_played=[games]




all_max_points=[]


inputs = list(itertools.product(num_subjects,num_trials,omega1,bandits,rew_func,games_played))

average_rewards=[]

start=time.time()
print('here')
if __name__=='__main__':
	with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
		result = executor.map(simulate_VKF_binary,inputs)
		res=[r for r in result]
		


print('time to finish : {}'.format(time.time()-start))

# np.save('dangerfirst_dangenv_empiricaldata_order1a',average_rewards)
 


points_earned_average=[x[0] for x in res]
print('length of earned rewards :{}'.format(len(points_earned_average)))
stds=[x[1] for x in res]
std_range='min SE= {}, max SE: {}'.format(min(stds)/np.sqrt(games),max(stds)/np.sqrt(games))
std_average='average SE= {}'.format(np.mean(stds))
max_points=max(points_earned_average)
all_max_points.append(max_points)
index_points=points_earned_average.index(max_points)
print(points_earned_average)
print('max points: {} at index {} at learning_rate {} RF1 -> RF2 REG'.format(max_points,index_points,inputs[index_points][2]))
# print('max betas avg reward: {}'.format(eq[215]))
# print('')
print('{}'.format(std_range))
print('{}'.format(std_average))






# # beta3=np.arange(0,6,1)
# bandits=np.load('safe_firsta.npy')

# bandits=[bandits]

# rf2=np.load('rf_safe_80trials.npy')
# rf2=rf2/3.0
# rf1=np.load('rf_pred_80trials.npy')
# rf1=rf1/3.0

# rew_func=np.concatenate((rf2,rf1),axis=0)

# rew_func=[rew_func]
# games=5000
# games_played=[games]
# inputs = list(itertools.product(num_subjects,num_trials,lrf,lrv,beta_safe,beta_pred,mf_betas,lr_cfs,sticks,pred_changes,rew_changes,rfi_rews,rfi_preds,bandits,rew_func,games_played))

# average_rewards=[]

# start=time.time()
# if __name__=='__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results=0
#         results = executor.map(simulate_VKF_binary,inputs)
#         res=[r for r in results]
#         print(res)
#         average_rewards.append(res)
# print('time to finish : {}'.format(time.time()-start))

# eqs=list(average_rewards[0])
# points_earned_average=[x[0] for x in eqs]
# print('length of earned rewards :{}'.format(len(points_earned_average)))
# stds=[x[1] for x in eqs]
# std_range='min SE= {}, max SE: {}'.format(min(stds)/np.sqrt(games),max(stds)/np.sqrt(games))
# std_average='average SE= {}'.format(np.mean(stds))
# max_points=max(points_earned_average)
# all_max_points.append(max_points)
# index_points=points_earned_average.index(max_points)
# print(points_earned_average)
# print('max points: {} at index {} at learning_rate {} RF2 -> RF1 REG'.format(max_points,index_points,lrf[index_points]))
# # print('max betas avg reward: {}'.format(eq[215]))
# # print('')
# print('{}'.format(std_range))
# print('{}'.format(std_average))





# # beta3=np.arange(0,6,1)
# bandits=np.load('safe_firsta.npy')

# bandits=[bandits]

# rf2=np.load('rf_safe_80trials.npy')
# rf2=rf2/3.0
# rf1=np.load('rf_pred_80trials.npy')
# rf1=rf1/3.0

# rew_func=np.concatenate((rf2,rf1),axis=0)
# rew_func=rew_func*-1

# rew_func=[rew_func]
# games=5000
# games_played=[games]

# inputs = list(itertools.product(num_subjects,num_trials,lrf,lrv,beta_safe,beta_pred,mf_betas,lr_cfs,sticks,pred_changes,rew_changes,rfi_rews,rfi_preds,bandits,rew_func,games_played))

# average_rewards=[]

# start=time.time()
# if __name__=='__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results=0
#         results = executor.map(simulate_VKF_binary,inputs)
#         res=[r for r in results]
#         print(res)
#         average_rewards.append(res)
# print('time to finish : {}'.format(time.time()-start))

# eqs=list(average_rewards[0])
# points_earned_average=[x[0] for x in eqs]
# print('length of earned rewards :{}'.format(len(points_earned_average)))
# stds=[x[1] for x in eqs]
# std_range='min SE= {}, max SE: {}'.format(min(stds)/np.sqrt(games),max(stds)/np.sqrt(games))
# std_average='average SE= {}'.format(np.mean(stds))
# max_points=max(points_earned_average)
# all_max_points.append(max_points)
# index_points=points_earned_average.index(max_points)
# print(points_earned_average)
# print('max points: {} at index {} at learning_rate {} RF2 -> RF1 FLIPPED'.format(max_points,index_points,lrf[index_points]))
# # print('max betas avg reward: {}'.format(eq[215]))
# # print('')
# print('{}'.format(std_range))
# print('{}'.format(std_average))




# # beta3=np.arange(0,6,1)
# bandits=np.load('safe_firsta.npy')

# bandits=[bandits]

# rf2=np.load('rf_safe_80trials.npy')
# rf2=rf2/3.0
# rf1=np.load('rf_pred_80trials.npy')
# rf1=rf1/3.0

# rew_func=np.concatenate((rf1,rf2),axis=0)
# rew_func=rew_func*-1

# rew_func=[rew_func]
# games=5000
# games_played=[games]

# inputs = list(itertools.product(num_subjects,num_trials,lrf,lrv,beta_safe,beta_pred,mf_betas,lr_cfs,sticks,pred_changes,rew_changes,rfi_rews,rfi_preds,bandits,rew_func,games_played))

# average_rewards=[]

# start=time.time()
# if __name__=='__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results=0
#         results = executor.map(simulate_VKF_binary,inputs)
#         res=[r for r in results]
#         print(res)
#         average_rewards.append(res)
# print('time to finish : {}'.format(time.time()-start))

# eqs=list(average_rewards[0])
# points_earned_average=[x[0] for x in eqs]
# print('length of earned rewards :{}'.format(len(points_earned_average)))
# stds=[x[1] for x in eqs]
# std_range='min SE= {}, max SE: {}'.format(min(stds)/np.sqrt(games),max(stds)/np.sqrt(games))
# std_average='average SE= {}'.format(np.mean(stds))
# max_points=max(points_earned_average)
# all_max_points.append(max_points)
# index_points=points_earned_average.index(max_points)
# print(points_earned_average)
# print('max points: {} at index {} at learning_rate {} RF21-> RF2 FLIPPED'.format(max_points,index_points,lrf[index_points]))
# # print('max betas avg reward: {}'.format(eq[215]))
# # print('')
# print('{}'.format(std_range))
# print('{}'.format(std_average))

# print('')
# print('AVERAGE ALL ORDERS : {}'.format(np.mean(all_max_points)))