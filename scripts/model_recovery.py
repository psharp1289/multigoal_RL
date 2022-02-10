#!/usr/bin/env python
# coding: utf-8

# Scripts used to fit models to Multigoal RL task
# Structure of meta script
#    1. Simulation scripts to generate data based off of different models
#    2. Likelihood functions


#  Iterate over synthetic data and model-fitting 10 times
#    3. Generate synthetic data
#    4. Build models to be fit to empirical data
#    5. Hierarchical model fitting functions
#    6. Fit each model to the empirical data


# We used a hierarchical Expectation-Maximization procedure called iterative importance sampling to fit the data
# 
# This algorithm iteratively computes iBIC via a sampling procedure. Each iteration of the loop a posterior sample for each subject is derived by 
# weighting the distribution of parameters by their likelihood. These samples at the subject level are concatenated, and then hyperparameters are fit 
# to these distributions. Model evidence at the group level is a sum of subject-level evidence (given that is in log space). 
# Once model evidence (defined by iBIC) does not improve, the iterations terminate. 
# 
# For each subject
# 
# 1. Sample from a prior distribution over $\forall i$ paramters $\theta_{i}$
# 
# 2. Gennerate likelihoods $p(data|\theta_i)$
# 
# 3. Compute posterior distribution over parameters at individual level by $p(\theta)\cdot p(data|\theta)$
# 
# For group:
# 
# 4. Fit new hyperparameters to the full sample (concatenate all individual-level posteriors) then go back to (1) unless iBIC hasn't improved



import os



################################################################## SIMULATION SCRIPTS ##############################################################






def simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st,safe_first):
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp
#     from sklearn.preprocessing import MinMaxScaler as scaler
	
	index=0

	for bandit in bandits:
		exec('global bandit_{}; bandit_{}=bandit'.format(index,index))
		index+=1
	num_bandits=len(bandits) 
	#define hyperparameters for each group distribution and sample from them
	all_lr_values=lrv
	all_lr_features=lrf
	all_datasets=[]
	all_outcomes=[]
	q_rew_all=[]
	
	reward_function_time_series=rew_func
   
	#retrieve samples of parameters from model
	for subject in range(num_subjects):
		print(subject)
		for game_played in range(1):
			exec('dataset_sub_{}_game_{}=pd.DataFrame()'.format(subject,game_played))
			#initialize variables for each subjects' data
			q_values=np.zeros((num_bandits,)) #initialize values for each sample, assumes 2 choices
			last_choice=np.zeros((num_bandits,))
			feature_matrix=np.zeros((2,2))
			lrf0=all_lr_features[0][subject]
			lrf1=all_lr_features[1][subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cfr[subject]
			lr_cf_p=lr_cfp[subject]
			last_beta=st[subject]
			lr_value=all_lr_values[subject]
			
			#collect subject level beta weights
			beta_safe=betas[0][subject]
			beta_pred=betas[1][subject]
			pred_changes=pred_changer[subject]
			rew_changes=rew_changer[subject]
			# betas_rew=np.array((beta_rew[subject],beta_rew[subject]))
			# betas_punish=np.array((beta_punish[subject],beta_punish[subject]))
			beta_weights_SR=np.array((beta_safe,beta_pred))
			beta_mf=mf_betas[subject]
			rfi_rew=rfi_rews[subject]
			rfi_pred=rfi_preds[subject]
			switching=[]
			choices=[]
			other_choices=[]
			outcomes=[]
			q_rews=[]
			safes=[]
			preds=[]


			#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
			for trial in range(num_trials):
				if safe_first==1:
					if trial>=80:
						beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
						
					else:
						beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))
				else:
					if trial<80:
						beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
						
					else:
						beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))



				# beta_weights_SR=np.array((beta_safe,beta_pred))

						
				# participant is told which predator they're facing
				current_reward_function=reward_function_time_series[trial]

				#weighting is a combo of beta weights and reward function
				full_weights=beta_weights_SR*current_reward_function
				#derive SR q-values by multiplying feature vectors by reward function
				q_sr=np.matmul(feature_matrix,full_weights)

				q_irf_weights=np.array((rfi_rew,rfi_pred))
				q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF +((last_beta)*(last_choice))
				# q_integrated=q_sr+((beta_mf)*(q_values))
				#determine choice via softmax
				action_values = np.exp((q_integrated)-sf.logsumexp(q_integrated));
				values=action_values.flatten()
				draw = choice(values,1,p=values)[0]
				indices = [i for i, x in enumerate(values) if x == draw]
				current_choice=sample(indices,1)[0]
				
				#save if participant switched
				if len(choices)==0:
					switching.append(0)
				else:
					if current_choice==choices[trial-1]-1:
						switching.append(0)
					else:
						switching.append(1)
						
				#save choice   
				choices.append(current_choice+1)
			 
				#get current feature latent probabilities
				
				cb1=[x[trial] for x in bandit_0]
				cb2=[x[trial] for x in bandit_1]

				current_bandit=np.array((cb1,cb2))
#                     print(np.random.binomial(1,[current_bandit][0]))
				#get feature outcomes for current trial
		   
				feature_outcomes=np.random.binomial(1,current_bandit)
				safes.append(list(feature_outcomes[:,0]))                    
				preds.append(list(feature_outcomes[:,1])) 
				
				#concatenate all feature outcomes into single array
				current_features=feature_outcomes[current_choice]
				
				#determine current outcome by feature vector * reward function
				current_outcome=sum(current_features*current_reward_function)

				#save how distant estimated EV is from actual EV 
				q_rews.append(np.abs(current_bandit-(q_sr[current_choice])))
				outcomes.append(current_outcome)
				
				# Prediction error over features
				pe_features = feature_outcomes-feature_matrix # Feature prediction errors
				#Feature updating
				current_decision=current_choice
				last_choice=np.zeros((num_bandits,))
				last_choice[current_decision]=1.0

				other_decision=abs(current_choice-1)
				other_choices.append(other_decision+1)
				#safe feature
				feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lr_features[0]*pe_features[current_decision,0])
				feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf_r*pe_features[other_decision,0])
				#predator feature updating
				feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lr_features[1]*pe_features[current_decision,1])
				feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf_r*pe_features[other_decision,1])

				#Model-Free action value updating
				q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))

			#save all output
			q_rew_all.append(np.sum(q_rews))
			all_outcomes.append(np.sum(outcomes))

			exec('dataset_sub_{}_game_{}["rf"]=list(reward_function_time_series)'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["sf"]=safe_first'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds





def simulate_one_step_featurelearner_counterfactual_2states_1iRF_sticky(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st,safe_first):
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp
#     from sklearn.preprocessing import MinMaxScaler as scaler
	
	index=0

	for bandit in bandits:
		exec('global bandit_{}; bandit_{}=bandit'.format(index,index))
		index+=1
	num_bandits=len(bandits) 
	#define hyperparameters for each group distribution and sample from them
	all_lr_values=lrv
	all_lr_features=lrf
	all_datasets=[]
	all_outcomes=[]
	q_rew_all=[]
	
	reward_function_time_series=rew_func
   
	#retrieve samples of parameters from model
	for subject in range(num_subjects):
		print(subject)
		for game_played in range(1):
			exec('dataset_sub_{}_game_{}=pd.DataFrame()'.format(subject,game_played))
			#initialize variables for each subjects' data
			q_values=np.zeros((num_bandits,)) #initialize values for each sample, assumes 2 choices
			last_choice=np.zeros((num_bandits,))
			feature_matrix=np.zeros((2,2))
			lrf0=all_lr_features[0][subject]
			lrf1=all_lr_features[1][subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cfr[subject]
			lr_cf_p=lr_cfp[subject]
			last_beta=st[subject]
			lr_value=all_lr_values[subject]
			
			#collect subject level beta weights
			beta_safe=betas[0][subject]
			beta_pred=betas[1][subject]
			pred_changes=pred_changer[subject]
			rew_changes=rew_changer[subject]
			# betas_rew=np.array((beta_rew[subject],beta_rew[subject]))
			# betas_punish=np.array((beta_punish[subject],beta_punish[subject]))
			beta_weights_SR=np.array((beta_safe,beta_pred))
			beta_mf=mf_betas[subject]
			rfi_rew=rfi_rews[subject]
			rfi_pred=rfi_rews[subject]
			switching=[]
			choices=[]
			other_choices=[]
			outcomes=[]
			q_rews=[]
			safes=[]
			preds=[]


			#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
			for trial in range(num_trials):
				if safe_first==1:
					if trial>=80:
						beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
						
					else:
						beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))
				else:
					if trial<80:
						beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
						
					else:
						beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))



				# beta_weights_SR=np.array((beta_safe,beta_pred))

						
				# participant is told which predator they're facing
				current_reward_function=reward_function_time_series[trial]

				#weighting is a combo of beta weights and reward function
				full_weights=beta_weights_SR*current_reward_function
				#derive SR q-values by multiplying feature vectors by reward function
				q_sr=np.matmul(feature_matrix,full_weights)

				q_irf_weights=np.array((rfi_rew,rfi_pred))
				q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF +((last_beta)*(last_choice))
				# q_integrated=q_sr+((beta_mf)*(q_values))
				#determine choice via softmax
				action_values = np.exp((q_integrated)-sf.logsumexp(q_integrated));
				values=action_values.flatten()
				draw = choice(values,1,p=values)[0]
				indices = [i for i, x in enumerate(values) if x == draw]
				current_choice=sample(indices,1)[0]
				
				#save if participant switched
				if len(choices)==0:
					switching.append(0)
				else:
					if current_choice==choices[trial-1]-1:
						switching.append(0)
					else:
						switching.append(1)
						
				#save choice   
				choices.append(current_choice+1)
			 
				#get current feature latent probabilities
				
				cb1=[x[trial] for x in bandit_0]
				cb2=[x[trial] for x in bandit_1]

				current_bandit=np.array((cb1,cb2))
#                     print(np.random.binomial(1,[current_bandit][0]))
				#get feature outcomes for current trial
		   
				feature_outcomes=np.random.binomial(1,current_bandit)
				safes.append(list(feature_outcomes[:,0]))                    
				preds.append(list(feature_outcomes[:,1])) 
				
				#concatenate all feature outcomes into single array
				current_features=feature_outcomes[current_choice]
				
				#determine current outcome by feature vector * reward function
				current_outcome=sum(current_features*current_reward_function)

				#save how distant estimated EV is from actual EV 
				q_rews.append(np.abs(current_bandit-(q_sr[current_choice])))
				outcomes.append(current_outcome)
				
				# Prediction error over features
				pe_features = feature_outcomes-feature_matrix # Feature prediction errors
				#Feature updating
				current_decision=current_choice
				last_choice=np.zeros((num_bandits,))
				last_choice[current_decision]=1.0

				other_decision=abs(current_choice-1)
				other_choices.append(other_decision+1)
				#safe feature
				feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lr_features[0]*pe_features[current_decision,0])
				feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf_r*pe_features[other_decision,0])
				#predator feature updating
				feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lr_features[1]*pe_features[current_decision,1])
				feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf_r*pe_features[other_decision,1])

				#Model-Free action value updating
				q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))

			#save all output
			q_rew_all.append(np.sum(q_rews))
			all_outcomes.append(np.sum(outcomes))

			exec('dataset_sub_{}_game_{}["rf"]=list(reward_function_time_series)'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["sf"]=safe_first'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds




####################################################################### LIKELIHOOD FUNCTIONS #########################################################################


def feature_learner_lik_beta_counterfactual_two(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	
	lik=np.zeros((sample_size,))
	lr_cf=samples[5][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]

	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	
	for index in range(len(choice)):
	
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
	   

		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)

		
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_change(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]

	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	
	for index in range(len(choice)):
		if index>=80:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
		else:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		# all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
	   

		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)

		
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_change_iRF(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	beta_rfinsensitive=samples[8][rng_samples]
	betas_rfinsensitive=beta_rfinsensitive.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	
	for index in range(len(choice)):
		if index>=80:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
		else:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		# all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
	   

		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)

		
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)
		rf_insensitive_weights=betas_rfinsensitive*np.array((1.0,-1.0))
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_change_2iRF(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	
	for index in range(len(choice)):
		if index>=80:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
		else:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		# all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
	   

		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)

		
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)


		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)), axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik



def feature_learner_lik_beta_counterfactual_two_change_2iRF_sticky(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	sf=data.sf[0]
	
	for index in range(len(choice)):
		if sf==1:
			if index>=80:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		else:
			if index<80:
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


		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)), axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi+(last_choice*beta_last)
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_change_1iRF_sticky(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	rfi_beta=samples[8][rng_samples]

	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	beta_last=samples[9][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	sf=data.sf[0]
	
	for index in range(len(choice)):
		if sf==1:
			if index>=80:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		else:
			if index<80:
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


		rf_insensitive_weights=np.concatenate((rfi_beta.reshape(sample_size,1),rfi_beta.reshape(sample_size,1)), axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi+(last_choice*beta_last)
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik





############################################### DEFINE FUNCTIONS TO BE USED IN HIERARCHICAL MODELLING ###############################################################


#function to sample parameters within a model

def sample_parameters(distribution_type,hyperparameter_list,sample_size):
	from numpy.random import beta,gamma,chisquare,poisson,uniform,logistic,multinomial,binomial
	from numpy.random import normal as norm
	counter=1
	for num in hyperparameter_list:
		exec("global hp_{}; hp_{}={}".format(counter,counter,num))
		counter+=1
	exec("global sample; sample={}({},{},{})".format(distribution_type,hp_1,hp_2,sample_size))

	return sample
	
# MODEL CLASS: assumes a hierarchical model 
# population parameters and subject level parameters 
# are jointly fit

#  The structure of the model class:
#         GROUP LEVEL:
#              Name - E.g., Standard Q Learning
#              Sample Size - How many samples for each parameter
#              Lik - Likelihood function
#              iBIC - Total evidence accounting for model complexity
#              Total_Evidence: Sum of Subject-Level Evidences (sum b/c in log-space)
#              Parameters (entries below x #parameters):
#                     Hyperparameters - e.g., [mu,sigma]
#                     Distribution Type - e.g., Beta
#                     Name - e.g., Lrate-Value
#
#        SUBJECT LEVEL: 
#           Evidence (i.e., log mean likelihood)
#           Parameters (entries below x #parameters):       
#                 posterior mean
#                 credible interval (95%)
#                 samples
#                 num_good_samples (not non or inf samples)

   # import the types


class model:
	
	def __init__(self):
		self.name=0
		self.num_subjects=0
		self.params=self.groupParams()
		self.subjfit=self.subjFit()
		self.subj_level_info=self.subjFit.subjParams()
		self.sample_size=0
		self.model_evidence_group=0
		self.bic=10**10 #arbitrarily high starting iBIC
		self.lik_func=0 #likelihood function
		
	
	class groupParams:
		
		def __init__(self):
			self.name='eta'
			self.distribution='beta'
			self.hyperparameters=[1,2]
							
	class subjFit:
		def __init__(self):
			self.model_evidence_subject=0
			self.subj_level_info=self.subjParams()
			
		class subjParams:
			def __init__(self):
				self.posterior_mean=0
				self.credible_interval=0
				self.samples=[]
				self.num_good_samples=[] #not nan or inf
				

#retrieve list of parameters from model
def get_parameters_for_model(model):
	parameters=[]
	parameter_dict={}
	for var in vars(model):
		exec('global x; x={}.{}'.format(model.name,var))
		if type(x)==model.groupParams:
			if var!='params':
				parameters.append(var)

	return parameters

def get_parameters_for_model_parallel(model):
	parameters=[]

	for var in vars(model):
		exec('global x; x={}.{}'.format(model.name,var))
		if type(x)==model.groupParams:
			if var!='params':
				param_info=[]
				param_info.append(var)
				exec('param_info.append({}.{}.distribution)'.format(model.name,var))
				exec('param_info.append({}.{}.hyperparameters)'.format(model.name,var))
				parameters.append(param_info)

	return parameters

def build_model(name,likelihood,group_parameters_info,number_subjects,sample_size):
	from scipy.stats import beta,gamma,norm,poisson,uniform,logistic
	#  INPUTS:
	#     name = name of model
	#     likelihood = likelihood function
	#     group_parameter_info = *Python dictionary* 
	#       defining parameter names, distributions and hyperparameters
	#       EXAMPLE: {'eta':['beta',[1,1]]}
	#     sample_size = number of samples from prior over group parameters
	
	#  OUTPUTS:
	#     model class (see above)
	
	exec('{}=model()'.format(name,name),globals())
	exec('{}.name="{}"'.format(name,name))
	exec('{}.num_subjects={}'.format(name,number_subjects))
	exec('{}.lik_func={}'.format(name,likelihood))
	exec('{}.sample_size={}'.format(name,sample_size))
	
	#encode in model the number of subjects and parameters in one's data
	for parameter in group_parameters_info:
		exec('{}.{}={}.groupParams()'.format(name,parameter,name))
		exec('{}.{}.name="{}"'.format(name,parameter,parameter))
		exec('{}.{}.distribution="{}"'.format(name,parameter,group_parameters_info[parameter][0]))
		exec('{}.{}.hyperparameters={}'.format(name,parameter,group_parameters_info[parameter][1]))
		exec('{}.{}.sample_size={}'.format(name,parameter,sample_size))
		exec('{}.{}.samples=sample_parameters("{}",{},{})'.format(name,parameter,group_parameters_info[parameter][0],group_parameters_info[parameter][1],sample_size))

	for sub in range(number_subjects):
		exec('{}.subject{}={}.subjFit()'.format(name,sub,name))
		for parameter in group_parameters_info:
			exec('{}.subject{}.{}={}.subject{}.subjParams()'.format(name,sub,parameter,name,sub))

def each_param(parameter,model,valid_parameter_indices,weights,subject,good_samples):
	exec('global parameter_samples; parameter_samples={}.{}.samples'.format(model.name,parameter))
	parameter_samps=np.reshape(parameter_samples,(model.sample_size,1))
	good_parameters=parameter_samps[valid_parameter_indices]

	mean,ci,sample=compute_samples(good_parameters,weights)
	exec('{}.subject{}.{}.posterior_mean={}'.format(model.name,subject,parameter,mean))
	exec('{}.subject{}.{}.credible_interval={}'.format(model.name,subject,parameter,ci))
	exec('{}.subject{}.{}.samples={}'.format(model.name,subject,parameter,sample))
	exec('{}.subject{}.{}.num_good_samples={}'.format(model.name,subject,parameter,good_samples))

from numpy.random import choice
import random
def each_param_parallel(parameter_samples,valid_parameter_indices,weights):
	parameter_samps=np.array(parameter_samples)
	good_parameters=parameter_samps[valid_parameter_indices]
	df=pd.DataFrame()
	weights=np.divide(weights,np.sum(weights))
	df['weights']=weights
	df['parameter_samples']=parameter_samps    
	df2=df.sort_values('parameter_samples')
	df2=df2.reset_index(drop=True)
	mean=np.sum(df['weights']*df['parameter_samples'])
	cdf=df2.weights.cumsum()
	cdf=np.array(cdf)
	samples_unweighted=list(df2['parameter_samples'])
	likelihood_weights=list(df2['weights'])
	samples = random.choices(samples_unweighted,cum_weights=list(cdf),k=10000)
	samples=list(samples)
	index_5=next(x[0] for x in enumerate(cdf) if x[1] > 0.0499999)
	index_95=next(x[0] for x in enumerate(cdf) if x[1] > 0.9499999) 
	ci_lower=df2['parameter_samples'][index_5]
	ci_higher=df2['parameter_samples'][index_95]
	ci=[ci_lower,ci_higher]
	results=[mean,ci,samples]
	return results


##from numba import vectorize, float64


def derive_posterior_samples(likelihood_vector,subject):    
	not_infs= ~np.isinf(likelihood_vector)
	not_nans= ~np.isnan(likelihood_vector)
	valid_parameter_indices=not_infs==not_nans     

	likelihood_vector_noinf=likelihood_vector[~np.isinf(likelihood_vector)] 
	likelihood_vector_cleaned=likelihood_vector_noinf[~np.isnan(likelihood_vector_noinf)] 
	good_samples=len(likelihood_vector_cleaned) 

	weights=np.exp(likelihood_vector_cleaned) 
	
	return likelihood_vector_cleaned,valid_parameter_indices,weights,good_samples
	
		


def compute_samples(parameter_samples,weights):
	import time
	import pandas as pd
	
	indices=np.array(indices)
	samples=df2['parameter_samples'][indices]
	samples=list(samples)
	index_5=next(x[0] for x in enumerate(cdf) if x[1] > 0.0499999) 
	index_95=next(x[0] for x in enumerate(cdf) if x[1] > 0.9499999) 
	ci_lower=df2['parameter_samples'][index_5]
	ci_higher=df2['parameter_samples'][index_95]
	ci=[ci_lower,ci_higher]
	
	return val,ci,samples


#fit hyperparameters to group sampled values from posterior. Be sure that you make sure output
# is appropriately handled for non-traditional (i.e., not beta, gamma, normal) distribution


def fit_hyperparameters(model):
	from scipy.stats import beta,gamma,norm,poisson,uniform,logistic
	parameters=get_parameters_for_model(model)
	number_subjects=model.num_subjects
	model_name=model.name
	for parameter in parameters:
		parameter_full_sample=[]
		exec('global distribution; distribution={}.{}.distribution'.format(model_name,parameter))
		for subject in range(number_subjects):
			exec('parameter_full_sample+={}.subject{}.{}.samples'.format(model_name,subject,parameter))
		exec('global hyperparameters; hyperparameters={}.fit(parameter_full_sample)'.format(distribution))
		if distribution=='gamma':
			h1=hyperparameters[0]
			h2=hyperparameters[2]
		elif distribution=='uniform':
			h1=hyperparameters[0]
			h2=hyperparameters[1]+hyperparameters[0]
		else:
			h1=hyperparameters[0]
			h2=hyperparameters[1]
		exec('{}.{}.hyperparameters={}'.format(model.name,parameter,[h1,h2]))



def fit_hyperparameters_parallel(model,parameter_info,all_results,number_subjects):
	from scipy.stats import beta,gamma,norm,poisson,uniform,logistic
	for parameter in parameter_info:
		parameter_full_sample=[]
		for item in all_results:
			for items in item:
				if parameter[0] in items:
					parameter_full_sample+=items[3]

		
		if parameter[1]=='gamma':
			hyperparameters=gamma.fit(parameter_full_sample,floc=0)
			h1=hyperparameters[0]
			h2=hyperparameters[2]
		elif parameter[1]=='uniform':
			hyperparameters=uniform.fit(parameter_full_sample)
			h1=hyperparameters[0]
			h2=hyperparameters[1]+hyperparameters[0]
		elif parameter[1]=='norm':
			hyperparameters=norm.fit(parameter_full_sample)
			h1=hyperparameters[0]
			h2=hyperparameters[1]
		else:
			hyperparameters=beta.fit(parameter_full_sample,floc=0,fscale=1)
			h1=hyperparameters[0]
			h2=hyperparameters[1]
		exec('{}.{}.hyperparameters={}'.format(model.name,parameter[0],[h1,h2]))


def sample_group_distributions(model):

	parameters=get_parameters_for_model(model)
	number_subjects=model.num_subjects
	model_name=model.name
	for parameter in parameters:
		exec('global distribution; distribution={}.{}.distribution'.format(model_name,parameter))
		exec('global hyperparameters; hyperparameters={}.{}.hyperparameters'.format(model_name,parameter))
		exec('{}.{}.samples=sample_parameters("{}",{},{})'.format(model_name,parameter,distribution,hyperparameters,model.sample_size))


def sample_group_distributions_parallel(params,sample_size):
	all_parameter_samples=[]
	for parameter in params:
		param=parameter[0]
		distribution=parameter[1]
		hyperparameters=parameter[2]
		samples=sample_parameters(distribution,hyperparameters,sample_size)
		all_parameter_samples.append(samples)
	return all_parameter_samples



def get_total_evidence(model):
	number_subjects=model.num_subjects
	model_name=model.name
	group_model_evidence=0
	for subject in range(number_subjects):
		exec('global subjEvidence; subjEvidence={}.subject{}.model_evidence_subject'.format(model_name,subject))
		group_model_evidence+=subjEvidence
	return group_model_evidence



def process_subject(subject,parameter_info,all_data,lik_func,parameter_sample_size,samples_partitioned,cores):
		data=all_data[subject]
		data=data.reset_index()
		parameter_names=[x[0] for x in parameter_info]
		samples_a=sample_group_distributions_parallel(parameter_info,parameter_sample_size)

		inputs=zip(repeat(samples_a),repeat(data),samples_partitioned)
		rng=np.arange(0,parameter_sample_size,1)
		
		likelihood=lik_func(samples_a,data,rng)
		likelihood=np.array(likelihood)

		#parallelize likelihood
		# if __name__=='__main__':
		#     pool = Pool(processes=cores)
		#     results=pool.starmap(lik_func, inputs)
		#     pool.close()
		#     pool.join()
		#     likelihood = [item for sublist in results for item in sublist]


		likelihood_vector_cleaned,valid_parameter_indices,weights,good_samples=derive_posterior_samples(likelihood,subject)
		# log mean likelihood
		model_evidence=scipy.special.logsumexp(likelihood_vector_cleaned,axis=0)-np.log(good_samples)
		#save  data
		subject_name='subject_{}'.format(subject)
		#return_dict_info=[[data.subjectID[10]]]
		return_dict_info=[[subject]]
		return_dict_info.append([model_evidence])
		#resample parameters and finish saving data
		# counter=0
		# inputs=zip(samples_a,repeat(valid_parameter_indices),repeat(weights))
		# if __name__=='__main__':
		#     pool = Pool(processes=len(parameter_names))
		#     results=pool.starmap(each_param_parallel, inputs)
		#     pool.close()
		#     pool.join()
		counter=0
		#resampling
		for param in parameter_names:
			new_samples=each_param_parallel(samples_a[counter],valid_parameter_indices,weights)
			ep=[]
			ep.append(param)
			ep.append(new_samples[0])
			ep.append(new_samples[1])
			ep.append(new_samples[2])
			# ep.append(results[counter][0])
			# ep.append(results[counter][1])
			# ep.append(results[counter][2])
			return_dict_info.append(ep)
			counter+=1

		return return_dict_info







#################################################### DEFINE MODEL RECOVERY VARIABLES ####################################################################
import pickle
import numpy as np

number_subjects=192
parameter_sample_size=100000

BICs_winning=[]
BICs_nonwinning=[]

x=np.arange(number_subjects)
cores_subs=12
bins=np.arange(0,number_subjects-1,number_subjects/cores_subs)
subjects_partitioned=[]
for i in range(1,cores_subs+1):
	subjects_partitioned.append(x[np.digitize(x,bins)==i])
print('subjects partitioned : {}'.format(subjects_partitioned))

#divide up likelihood over all cores
x=np.arange(parameter_sample_size)
cores=4
bins=np.arange(0,parameter_sample_size-1,parameter_sample_size/cores)
samples_partitioned=[]
for i in range(1,cores+1):
	samples_partitioned.append(x[np.digitize(x,bins)==i])
####################################################################### GENERATE SYNTHETIC DATA #########################################################
for i in range(9):
	# #### import pandas as pd
	import numpy as np
	from numpy.random import beta as betarnd
	from numpy.random import gamma as gammarnd
	from numpy.random import normal as normalrnd
	from random import sample
	from random import shuffle
	import pandas as pd
	import seaborn as sns
	# #assumes you've generated bandits variable with the block above
	df=pd.DataFrame()
	num_subjects=192
	num_trials=160
	lrv=list(betarnd(2.743734605320487, 2.0714203884252584,num_subjects))


	lrf_safe=list(betarnd(0.3238920704407903, 1.3271625781269714,num_subjects))
	lrf_pos=list(betarnd(0.6,2.0,num_subjects))
	lrf_neg=list(betarnd(0.4,1.2,num_subjects))

	pred_change=list(normalrnd(-0.14914530448854793, 1.0180861762150282,num_subjects))
	rew_change=list(normalrnd(0.5043001380934354, 1.1065215386632883,num_subjects))

	# lrf_mild=list(betarnd(2,5,num_subjects))
	lrf_each=list(betarnd(3,5,num_subjects))

	lr_cfr=list(betarnd(0.18712683635873772, 1.2398401783318014,num_subjects))
	lr_cfp=list(betarnd(0.40,1.2,num_subjects))

	

	betas=np.zeros((2,num_subjects))
	# beta_safe=list(gammarnd(0.75,6,num_subjects))
	# beta_pred=list(gammarnd(0.62,3.71,num_subjects))
	beta_safe=list(gammarnd(0.6025122100072803, 6.731039259186533,num_subjects))
	beta_pred=list(gammarnd(0.4529044061222983, 4.2339741652612215,num_subjects))
	betas[0]=beta_safe
	#np.zeros((num_subjects))+3.0 # learnabout safe feature
	# betas[1]=beta_mild # learn about mild feature (small reward)
	betas[1]=beta_pred
	lrf2=np.zeros((2,num_subjects))
	lrf2[0]=lrf_safe # learnabout safe feature
	lrf2[1]=lrf_safe # learnabout safe feature
	# lrf2[2]=lrf_safe # learnabout safe feature


	it=list(gammarnd(1,1,num_subjects))
	mf_betas=list(gammarnd(1.1093447925340514, 0.548557708547694,num_subjects))
	# beta_irf=list(gammarnd(0.29,3.92,num_subjects))
	rfi_rews=list(gammarnd(0.4275512172820504, 1.6876320240266436,num_subjects))
	rfi_preds=list(gammarnd(0.6019802760339684, 1.4308407307549147,num_subjects))
	rfi_total=list(gammarnd(0.53, 1.55,num_subjects))
	stick=list(normalrnd(0.10548073446686312, 0.5444583943396483,num_subjects))

	# mf_betas=np.zeros((num_subjects,1))
	bandits=np.load('safe_firsta.npy')
	rf2=np.load('rf_safe_80trials.npy')
	# rf2=np.flip(rf2,1)

	rf2=rf2/3.0
	rf1=np.load('rf_pred_80trials.npy')
	# rf1=np.flip(rf1,1)

	rf1=rf1/3.0

	safe_first=1
	safe_first1=0
	reward_function_timeseries=np.concatenate((rf2,rf1),axis=0)


	#CREATE SYNTHETIC DATA

	all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_total,rfi_total,stick,safe_first)
	# all_data4,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf22,lrv2,betas2,mf_betas2,bandits,reward_function_timeseries2,pred_change2,rew_change2,lr_cfr2,lr_cfp2,rfi_rews2,rfi_preds2,stick2,safe_first1)
	# all_data3=all_data3+all_data4
	# In[10]:
	import pandas as pd
	from itertools import repeat,starmap
	import numpy as np
	import time
	import scipy
	from numpy.random import beta,gamma,chisquare,normal,poisson,uniform,logistic,multinomial,binomial

	############################################################# BUILD MODELS TO BE FIT #########################################################

	

	all_models=[]


	name_1='feature_learner_winning_{}'.format(i)
	group_parameters_info_m={	'mfb':['gamma',[1, 1]],
								'lr_f':['beta',[1,1]],
								'pred_s':['gamma',[1, 1]],
								'pred_b':['gamma',[1, 1]],
								'lr_val':['beta',[1,1]],
								'lr_cf':['beta',[1,1]],							
								'pre_change':['norm',[0, 1]], 
								're_change':['norm',[0, 1]],
								'w_rfi_rew':['gamma',[1,1]],
								'w_rfi_prew':['gamma',[1,1]],
								'stick1':['norm',[0,1]]
							
								
								
								}
							 
	likelihood='feature_learner_lik_beta_counterfactual_two_change_2iRF_sticky'
	build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
	exec('all_models.append(feature_learner_winning_{})'.format(i))

	name_1='feature_learner_nonwinning_{}'.format(i)
	group_parameters_info_m={	'mfb':['gamma',[1, 1]],
								'lr_f':['beta',[1,1]],
								'pred_s':['gamma',[1, 1]],
								'pred_b':['gamma',[1, 1]],
								'lr_val':['beta',[1,1]],
								'lr_cf':['beta',[1,1]],							
								'pre_change':['norm',[0, 1]], 
								're_change':['norm',[0, 1]],
								'w_rfi_rew':['gamma',[1,1]],
								'stick1':['norm',[0,1]]
							
								
								
								}
							 
	likelihood='feature_learner_lik_beta_counterfactual_two_change_1iRF_sticky'
	build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
	exec('all_models.append(feature_learner_nonwinning_{})'.format(i+1))




	############################################################### FIT MODELS #############################################################

	import concurrent.futures
	from multiprocessing.dummy import Pool
	import multiprocessing
	from itertools import repeat,starmap
	import numpy as np
	import time


	#current_dataset=all_data_input #make sure dataset is correct before running model fitting!!
	current_dataset=all_data3
	models_BICs_record=[]
	#iterate over models
	counter=0
	for model in all_models:
		print(model.name)
		improvement=1 #arbitrary start to ensure while loop starts
		
		#keep sampling until no improvement in iBIC
		while improvement>0:
			
			#store old_bic for comparison to new_bic
			old_bic=model.bic

			
			#generate log likelihood for each subject and compute new samples
			procs = []
			parameter_info=get_parameters_for_model_parallel(model)
			print('current_hyperparms:')
			print(parameter_info)
			parameter_names=[x[0] for x in parameter_info]
			parameter_disributions=[x[1] for x in parameter_info]
			parameter_sample_size=model.sample_size
			subjects=list(np.arange(0,number_subjects,1))
		   
			lik_func=model.lik_func
			return_dict={}
			inputs=zip(subjects,repeat(parameter_info),repeat(current_dataset),
				repeat(lik_func),repeat(parameter_sample_size),repeat(samples_partitioned),repeat(cores))
			start_time = time.time()
			if __name__=='__main__':    
					pool = Pool(processes=cores_subs)
					results=pool.starmap(process_subject, inputs)
					pool.close()
					pool.join()
					
					print('total time: {}'.format(time.time()-start_time))
					exec('all_results_{} = [item for item in results]'.format(model.name))
					
			print('total time: {}'.format(time.time()-start_time))

			exec('all_results=all_results_{}'.format(model.name))
			#fit new hyperparameters from full posterior

			fit_hyperparameters_parallel(model,parameter_info,all_results,num_subjects)
			#compute iBIC
			Nparams = 2*len(parameter_names)
			print(Nparams)
			Ndatapoints = float(number_subjects*num_trials) #total number of datapoints
			exec('total_evidence=sum([x[1][0] for x in all_results_{}])'.format(model.name))
			new_bic = -2.0*float(total_evidence) + Nparams*np.log(Ndatapoints) # Bayesian Information Criterion
			improvement = old_bic - new_bic # compute improvement of fit
			
			#only retain evidence and BIC if they improve
			if improvement > 0:
				model.model_evidence_group=total_evidence
				model.bic=new_bic
			else:
				if 'feature_learner_nonwinning' in model.name:
					BICs_nonwinning.append(model.bic)
				else:
					BICs_winning.append(model.bic)

			
			#read out latest iteration
			print('{}- iBIC old:{}, new: {}\n'.format(model.name, old_bic, new_bic))
		models_BICs_record.append(model.bic)

	# import pickle
	# with open("iBICs_simples.txt", "a") as fp:   #Pickling
	# 	pickle.dump(models_BICs_record, fp)

num_recovered=0
for i in range(len(BICs_winning)):
	if BICs_winning[i]<BICs_nonwinning[i]:
		num_recovered+=1
print('')
print('Percent models accurately recovered: {}'.format(num_recovered/10.0))
print('')



