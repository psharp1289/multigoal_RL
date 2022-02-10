'''


'''
import concurrent.futures
import time
from itertools import repeat
from multiprocessing import Pool
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
'''
generate simulated data

'''
import itertools
import multiprocessing
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Generate values for each parameter
# inputs: num_subjects,num_trials,lrf,lrv,beta1,beta2,mf_betas,lr_cfs,sticks,pred_changes,rew_changes,rfi_rews,rfi_preds,bandits,rew_func,games_played


num_subjects=[1]
num_trials=[160]
lrf= np.arange(0,1.03,0.02)
lr_cfs=[0.13]
sticks=[0.10]
pred_changes=[-0.16]
rew_changes=[0.50]
rfi_rews=[0.68]
rfi_preds=[0.48]

# lrf=[0.2]
#lrf=[1]
lrv=[0.5]
mf_betas=[0.48]
beta_safe=[8.44] #np.arange(0)
beta_pred=[4.19] #np.arange(0)


# beta3=np.arange(0,6,1)
bandits=np.load('safe_firsta.npy')

bandits=[bandits]

rf2=np.load('rf_safe_80trials.npy')
rf2=rf2/3.0
rf1=np.load('rf_pred_80trials.npy')
rf1=rf1/3.0

rew_func=np.concatenate((rf2,rf1),axis=0)
# rew_func=rew_func*-1

# rew_func=np.flip((rew_func),axis=1)


# rew_func=rf1
# rew_func[:,[0, 1]] = rew_func[:,[1, 0]]
# rew_func=rew_func*-1.0
preds=[]
consec_preds=[]
sss=[]
consec_rews=[]
trajectory=[]
x_axis=np.arange(160)
c=0
consec=0
puns=0
rews=0
# for i in range(1,len(rew_func)):
	# if rew_func[i][0]!=rew_func[i-1][0]:
	# 	if rew_func[i][0]==1:
	# 		rews+=1
	# 	else:
	# 		puns+=1
for i in rew_func:
	if -1 in i:
		if c>0:
			if last_o==-1:
				consec+=1
			else:
				consec_rews.append(consec)
				consec=0
		last_o=-1
		preds.append(i)
		trajectory.append(-1)
		
	elif 1 in i:
		if c>0:
			if last_o==1:
				consec+=1
			else:
				consec_preds.append(consec)
				consec=0
		last_o=1
		sss.append(i)
		trajectory.append(1)
		
	c+=1
	
# print('number switches to pred {}'.format(puns))
# print('number switches to rews {}'.format(rews))

# lelelel
# for i in rew_func:
# 	if -1 in i:
# 		if c>0:
# 			if last_o==-1:
# 				consec+=1
# 			else:
# 				consec_rews.append(consec)
# 				consec=0
# 		last_o=-1
# 		preds.append(i)
# 		trajectory.append(-1)
		
# 	elif 1 in i:
# 		if c>0:
# 			if last_o==1:
# 				consec+=1
# 			else:
# 				consec_preds.append(consec)
# 				consec=0
# 		last_o=1
# 		sss.append(i)
# 		trajectory.append(1)
		
# 	c+=1


# print('number preds in RF: {}'.format(len(preds)))
# print('number safes in RF: {}'.format(len(sss)))

# print('average consec preds: {}'.format(np.mean(consec_preds)))
# print('average consec rews: {}'.format(np.mean(consec_rews)))


dataset=pd.DataFrame()
dataset['timepoint']=x_axis
dataset['rf']=trajectory
# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="rf",
             data=dataset)
plt.show()

rew_func=[rew_func]
games=5000
games_played=[games]


#Generate a list of tuples where each tuple is a combination of parameters.
#The list will contain all possible combinations of parameters.

# num_subjects=params[0]
# num_trials=params[1]
# lrf=params[2]
# lrv=params[3]
# beta_safe=params[4]
# beta_pred=params[5]
# beta_mf=params[6]
# lr_cf=params[8]
# last_beta=params[9]
# pred_changes=params[10]
# rew_changes=params[11]
# rfi_rew=params[12]
# rfi_pred=params[13]
# bandits=params[14]
# reward_function_time_series=params[15]
# games_played=params[16]

all_max_points=[]


inputs = list(itertools.product(num_subjects,num_trials,lrf,lrv,beta_safe,beta_pred,mf_betas,lr_cfs,sticks,pred_changes,rew_changes,rfi_rews,rfi_preds,bandits,rew_func,games_played))

average_rewards=[]

start=time.time()
if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=0
        results = executor.map(simulate_2iRF_sticky,inputs)
        res=[r for r in results]
        print(res)
        average_rewards.append(res)
print('time to finish : {}'.format(time.time()-start))


# np.save('dangerfirst_dangenv_empiricaldata_order1a',average_rewards)
 

eqs=list(average_rewards[0])
points_earned_average=[x[0] for x in eqs]
print('length of earned rewards :{}'.format(len(points_earned_average)))
stds=[x[1] for x in eqs]
std_range='min SE= {}, max SE: {}'.format(min(stds)/np.sqrt(games),max(stds)/np.sqrt(games))
std_average='average SE= {}'.format(np.mean(stds))
max_points=max(points_earned_average)
all_max_points.append(max_points)
index_points=points_earned_average.index(max_points)
print(points_earned_average)
print('max points: {} at index {} at learning_rate {} RF1 -> RF2 REG'.format(max_points,index_points,lrf[index_points]))
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
#         results = executor.map(simulate_2iRF_sticky_punishments,inputs)
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
#         results = executor.map(simulate_2iRF_sticky_pflipped,inputs)
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
#         results = executor.map(simulate_2iRF_sticky_pflipped,inputs)
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