'''


'''
import concurrent.futures
import time
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
'''
generate simulated data

'''
import itertools
import multiprocessing
for i in range(20):
	#Generate values for each parameter
	num_subjects=[1]
	num_trials=[80]
	lrf=[0.15]
	# lrf=[0.2]
	#lrf=[1]
	lrv=[0.0]
	mf_betas=[0.0]
	beta1=np.arange(6)
	beta2=np.arange(6)
	# beta3=np.arange(0,6,1)
	bandits=np.load('danger_first.npy')
	bandits=np.delete(bandits,np.arange(0,80),2)

	# stable probabilities
	bandits=np.zeros((2,2,80))
	for i in range(80):
		bandits[:,:,i]=[[0.4,0.6],[0.6,0.4]]
	# np.save('safe_second_block',bandits)
	bandits=[bandits]
	rf2=np.load('rf_safe_80trials.npy')
	rf2=rf2/3.0
	rf1=np.load('rf_pred_80trials.npy')
	rf1=rf1/3.0
	np.random.shuffle(rf1)
	# rew_func=np.concatenate((rf1,rf2),axis=0)
	rew_func=rf2
	rew_func=rew_func*-1.0
	rew_func=[rew_func]
	games_played=[500]
	#Generate a list of tuples where each tuple is a combination of parameters.
	#The list will contain all possible combinations of parameters.
	inputs = list(itertools.product(num_subjects,num_trials,lrf,lrv,beta1,beta2,mf_betas,bandits,rew_func,games_played))

	average_rewards=[]

	start=time.time()
	if __name__=='__main__':
	    with concurrent.futures.ProcessPoolExecutor() as executor:
	        results=0
	        results = executor.map(simulate_one_step_featurelearner_counterfactual_parallel_2states,inputs)
	        res=[r for r in results]
	        average_rewards.append(res)
	print('time to finish : {}'.format(time.time()-start))


	np.save('dangfirst_safeenv_rfsame_probabilitiesReversed_shuffle',average_rewards)
	 

	eqs=list(average_rewards[0])
	points_earned_average=[x[0] for x in eqs]
	stds=[x[1] for x in eqs]
	std_range='min SE= {}, max SE: {}'.format(np.min(stds)/np.sqrt(500),np.max(stds)/np.sqrt(500))
	std_average='average SE= {}'.format(np.mean(stds))
	min_points=min(points_earned_average)
	mean_points=np.mean(points_earned_average)
	# index_points=points_earned_average.index(mean_points)
	# print('mean points: {}, index: {}'.format(mean_points,index_points))
	# print('max betas avg reward: {}'.format(eq[215]))
	# print('')
	print('{}'.format(std_range))
	print('{}'.format(std_average))





	# Graph all simulations


	safe_rf=np.load('rf_pred_80trials.npy')
	safes=[]
	ppp=[]
	for i in safe_rf:
		if 3 in i:
			safes.append(i)
		elif -3 in i:
			ppp.append(i)
	# print('number safes in safe env: {}'.format(len(safes)))
	# print('number preds in safe env: {}'.format(len(ppp)))

	dang_rf=np.load('rf_safe_80trials.npy')
	preds=[]
	sss=[]
	for i in dang_rf:
		if -3 in i:
			preds.append(i)
		elif 3 in i:
			sss.append(i)
	# print('number preds in dang env: {}'.format(len(preds)))
	# print('number safes in dang env: {}'.format(len(sss)))

	# load environments
	safe_env=np.load('dangfirst_safeenv_rfsame_probabilitiesReversed_rfsReversed.npy')
	safe_env=safe_env[0,:,0]

	diffs_safe=[]
	diffs_dang=[]
	for i in range(len(safe_env)-6):
		diffs_safe.append(np.abs(safe_env[i]-safe_env[i+6]))

	skips=0
	for i in range(len(safe_env)):
		if (i+1)%6==0:
			x='skip'
			skips+=1
		else:
			diffs_dang.append(np.abs(safe_env[i]-safe_env[i+1]))

	print('skips: {}'.format(skips))
	print('SAFE ENV - mean difference due to safe beta weight = {}'.format(np.mean(diffs_safe)))
	print('SAFE ENV - mean difference due to danger beta weight = {}'.format(np.mean(diffs_dang)))

	dang_env=np.load('dangfirst_dangerenv_rfsame_probabilitiesReversed_rfsReversed.npy')
	dang_env=dang_env[0,:,0]

	diffs_safe=[]
	diffs_dang=[]
	for i in range(len(dang_env)-6):
		diffs_safe.append(np.abs(dang_env[i]-dang_env[i+6]))

	skips=0
	for i in range(len(dang_env)):
		if (i+1)%6==0:
			x='skip'
			skips+=1
		else:
			diffs_dang.append(np.abs(dang_env[i]-dang_env[i+1]))

	print('skips: {}'.format(skips))
	print('DANG ENV -mean difference due to safe beta weight = {}'.format(np.mean(diffs_safe)))
	print('DANG ENV -mean difference due to danger beta weight = {}'.format(np.mean(diffs_dang)))


	