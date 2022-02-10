from scipy import optimize
import numpy as np
from numba import jit
import os
from scipy.special import logsumexp
from numpy.random import beta,gamma,chisquare,poisson,uniform,logistic,multinomial,binomial
from numpy.random import normal as norm
import pandas as pd
import itertools
import multiprocessing
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize as mins
import concurrent.futures
import time
from itertools import repeat
from multiprocessing import Pool

os.chdir('../../feature_experiment/data')
#load in empirical data




# @jit(nopython=False)
def feature_learner_lik_beta_counterfactual_two_empirical_2lr(params):
	pc_m=params[0]
	pc_sd=params[1]
	rc_m=params[2]
	rc_sd=params[3]
	lrv_a=params[4]
	lrv_b=params[5]
	lrf_a=params[6]
	lrf_b=params[7]
	lrcf_a=params[8]
	lrcf_b=params[9]
	safe1=params[10]
	safe2=params[11]
	pred1=params[12]
	pred2=params[13]
	mf1=params[14]
	mf2=params[15]
	data=params[16]
	print(len(data))
	print('here')
	num_choices=2
	sample_size=1000
	lik=np.zeros((sample_size,))

	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=beta(lrcf_a,lrcf_b,sample_size)
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=norm(pc_m,pc_sd,sample_size)
	rew_changer=norm(rc_m,rc_sd,sample_size)
	lr_val=beta(lrv_a,lrv_b,sample_size)
	lr_features=beta(lrf_a,lrf_b,sample_size)
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=gamma(safe1,safe2,sample_size)
	betas_pred=gamma(pred1,pred2,sample_size)
	mf_betas=gamma(mf1,mf2,sample_size)
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
	data=data.reset_index()
	errors=data[data['chosen_state']=='SLOW']
	cut_beginning=0
	for i in errors.level_0:
		if i<81:
			cut_beginning+=1
	data = data[data['chosen_state']!='SLOW']
	data.dropna(subset=['chosen_state'], inplace=True)
	data=data.reset_index(drop=True)
	if data.safe_first[0]==True:
		sf=1
		
	else:
		sf=2
		
	#grab relevant data
	choice=data.chosen_state
	for i in data.version:
		version=i
	if version.endswith('b')==True:
		reverse='yes'
	else:
		reverse='no'

	if reverse=='no':
	 	reward_function_timeseries=[[float(data.f_1_reward[i]),float(data.f_2_reward[i])] for i in range(len(data.s_2_f_1_outcome))]
	 	outcome=data.reward_received
	 	safes=[[float(data.s_1_f_1_outcome[i]),float(data.s_2_f_1_outcome[i])] for i in range(len(data.s_2_f_1_outcome))]
	 	predators=[[float(data.s_1_f_2_outcome[i]),float(data.s_2_f_2_outcome[i])] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[float(data.f_2_reward[i]),float(data.f_1_reward[i])] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[float(data.s_1_f_2_outcome[i]),float(data.s_2_f_2_outcome[i])] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[float(data.s_1_f_1_outcome[i]),float(data.s_2_f_1_outcome[i])] for i in range(len(data.s_2_f_1_outcome))]

	for index in range(len(choice)):
		if sf==1:
			if index>=80-cut_beginning:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		else:
			if index<80-cut_beginning:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)
		q_integrated=q_sr+(q_values*mf_betas)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)
	
	lik=-1*lik #negative log likelihood is what we want to minimize
	return np.sum(lik)

def opt_group_level(parameters):
	subj_IDs=[pd.read_csv(x) for x in os.listdir(os.curdir) if x.startswith('sub_3')]
	pc_m,pc_sd,rc_m,rc_sd,lrv_a,lrv_b,lrf_a,lrf_b,lrcf_a,lrcf_b,safe1,safe2,pred1,pred2,mf1,mf2=parameters
	pcm=[pc_m]
	pcsd=[pc_sd]
	rcm=[rc_m]
	rcsd=[rc_sd]
	lrva=[lrv_a]
	lrvb=[lrv_b]
	lrfa=[lrf_a]
	lrfb=[lrf_b]
	lrcfa=[lrcf_a]
	lrcfb=[lrcf_b]
	safe1=[safe1]
	safe2=[safe2]
	pred1=[pred1]
	pred2=[pred2]
	mf1=[mf1]
	mf2=[mf2]


	print('mean pred change : {}, sd: {}'.format(pcm,pcsd))
	print('mean rew change : {}, sd: {}'.format(rcm,rcsd))
	inputs = list(itertools.product(pcm,pcsd,rcm,rcsd,lrva,lrvb,lrfa,lrfb,lrcfa,lrcfb,safe1,safe2,pred1,pred2,mf1,mf2,subj_IDs))

	with concurrent.futures.ProcessPoolExecutor() as executor:
		results=0
		results = executor.map(feature_learner_lik_beta_counterfactual_two_empirical_2lr,inputs)
		res=[r for r in results]
		average_rewards.append(res)

	eqs=list(average_rewards[0])
	likelihoods=[x[0] for x in eqs]
	return np.sum(likelihoods)

results=mins(opt_group_level,np.array([0.,1.,0.,1.,1,1,1,1,1,1,1,1,1,1,1,1]),
bounds=[(-5,5),(0,10),(-5,5),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10)],
method='trust-constr')
print(results.x)