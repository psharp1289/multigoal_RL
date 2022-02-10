#!/usr/bin/env python
# coding: utf-8

# Scripts used to fit models to Multigoal RL task
# Structure of meta script
#    1. Simulation scripts to generate data based off of different models
#    2. Generate synthetic data
#    3. Likelihood functions
#    4. Hierarchical model fitting functions
#    5. Loading in empirical data
#    6. Build models to be fit to empirical data
#    7. Fit each model to the empirical data
#    8. Save all fitted parameters and iBICs
#    9. IF simulated data, conduct parameter recovery

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
# 3. Compute posterior distribution over parameters at individual level by $p(\theta)\cdot p(data|\theta) via likelihood-weighted resampling$

# For group:
# 
# 4. Fit new hyperparameters to the full sample (concatenate all individual-level posteriors) then go back to (1) unless iBIC hasn't improved



import os



################################################################## SIMULATION SCRIPTS ##############################################################


def simulate_one_step_featurelearner_counterfactual_2states_nochange(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,lr_cfr):
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
		for game_played in range(1):
			exec('dataset_sub_{}_game_{}=pd.DataFrame()'.format(subject,game_played))
			#initialize variables for each subjects' data
			q_values=np.zeros((num_bandits,)) #initialize values for each sample, assumes 2 choices
			feature_matrix=np.zeros((2,2))

			lrf0=all_lr_features[0][subject]
			lrf1=all_lr_features[1][subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cfr[subject]
			lr_value=all_lr_values[subject]
			
			#collect subject level beta weights
			beta_safe=betas[0][subject]
			beta_pred=betas[1][subject]
			
			# betas_rew=np.array((beta_rew[subject],beta_rew[subject]))
			# betas_punish=np.array((beta_punish[subject],beta_punish[subject]))
			beta_weights_SR=np.array((beta_safe,beta_pred))
			beta_mf=mf_betas[subject]
			switching=[]
			choices=[]
			outcomes=[]
			q_rews=[]
			safes=[]
			predators=[]


			#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
			for trial in range(num_trials):

				beta_weights_SR=np.array((beta_safe,beta_pred))

						
				# participant is told which predator they're facing
				current_reward_function=reward_function_time_series[trial]


				#weighting is a combo of beta weights and reward function
				full_weights=beta_weights_SR*current_reward_function

				#derive SR q-values by multiplying feature vectors by reward function
				q_sr=np.matmul(feature_matrix,full_weights) 
				
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) 

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
					if current_choice==choices[trial-1]:
						switching.append(0)
					else:
						switching.append(1)
						
				#save choice   
				choices.append(current_choice+1)
			 
				#get current feature latent probabilities
				
				cb1=[x[trial] for x in bandit_0]
				cb2=[x[trial] for x in bandit_1]

				current_bandit=np.array((cb1,cb2))
				feature_outcomes=np.random.binomial(1,current_bandit)
				safes.append(list(feature_outcomes[:,0]))                    
				predators.append(list(feature_outcomes[:,1])) 
				
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
				other_decision=abs(current_choice-1)

				#safe feature
				feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lr_features[0]*pe_features[current_decision,0])
				feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf_r*pe_features[other_decision,0])
				#predator feature updating
				feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lr_features[1]*pe_features[current_decision,1])
				feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf_r*pe_features[other_decision,1])

				#Model-Free action value updating
				q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))

			exec('dataset_sub_{}_game_{}["rf"]=list(reward_function_time_series)'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["pred_o"]=predators'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,predators


def simulate_one_step_featurelearner_counterfactual_2states(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp):
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
			feature_matrix=np.zeros((2,2))

			lrf0=all_lr_features[0][subject]
			lrf1=all_lr_features[1][subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cfr[subject]
			lr_cf_p=lr_cfp[subject]
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
			switching=[]
			choices=[]
			outcomes=[]
			q_rews=[]
			safes=[]
			predators=[]


			#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
			for trial in range(num_trials):
				if trial>=80:
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
				
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) 

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
					if current_choice==choices[trial-1]:
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
				predators.append(list(feature_outcomes[:,1])) 
				
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
				other_decision=abs(current_choice-1)

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
			exec('dataset_sub_{}_game_{}["pred_o"]=predators'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,predators

def simulate_one_step_featurelearner_counterfactual_2states_peLR(num_subjects,num_trials,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cf,lr_neg,lr_pos):
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
			feature_matrix=np.zeros((2,2))

			lrf0=lr_neg[subject]
			lrf1=lr_pos[subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cf[subject]
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
			switching=[]
			choices=[]
			outcomes=[]
			q_rews=[]
			safes=[]
			predators=[]


			#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
			for trial in range(num_trials):
				if trial>=80:
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
				
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) 

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
					if current_choice==choices[trial-1]:
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
				predators.append(list(feature_outcomes[:,1])) 
				
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
				other_decision=abs(current_choice-1)

				#safe feature
				if feature_outcomes[current_decision,0]==0:
					feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lr_features[0]*pe_features[current_decision,0])
				elif feature_outcomes[current_decision,0]==1:
					feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lr_features[1]*pe_features[current_decision,0])
				
				#predator feature updating
				if feature_outcomes[current_decision,1]==0:
					feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lr_features[0]*pe_features[current_decision,1])
				elif feature_outcomes[current_decision,1]==1:
					feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lr_features[1]*pe_features[current_decision,1])


				feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf_r*pe_features[other_decision,0])
				feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf_r*pe_features[other_decision,1])

				#Model-Free action value updating
				q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))

			#save all output
			q_rew_all.append(np.sum(q_rews))
			all_outcomes.append(np.sum(outcomes))

			exec('dataset_sub_{}_game_{}["rf"]=list(reward_function_time_series)'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["pred_o"]=predators'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,predators

##### GENERATE SYNTHETIC DATA
def simulate_one_step_featurelearner_counterfactual_2states_iRF(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,betas_irf):
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
			feature_matrix=np.zeros((2,2))

			lrf0=all_lr_features[0][subject]
			lrf1=all_lr_features[1][subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cfr[subject]
			lr_cf_p=lr_cfp[subject]
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
			beta_irf=betas_irf[subject]
			switching=[]
			choices=[]
			other_choices=[]
			outcomes=[]
			q_rews=[]
			safes=[]
			preds=[]


			#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
			for trial in range(num_trials):
				if trial>=80:
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

				q_irf_weights=beta_irf*np.array((1.0,-1.0))
				q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF
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
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds

def simulate_one_step_featurelearner_counterfactual_2states_2iRF(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds):
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
			feature_matrix=np.zeros((2,2))

			lrf0=all_lr_features[0][subject]
			lrf1=all_lr_features[1][subject]
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=lr_cfr[subject]
			lr_cf_p=lr_cfp[subject]
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
				if trial>=80:
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
				q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF
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
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds

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


####################################################################### GENERATE SYNTHETIC DATA #########################################################

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
lrf=np.zeros((2,num_subjects))
lrf_safe=list(betarnd(0.3238920704407903, 1.3271625781269714,num_subjects))
lrf_pos=list(betarnd(0.6,2.0,num_subjects))
lrf_neg=list(betarnd(0.4,1.2,num_subjects))

pred_change=list(normalrnd(-0.14914530448854793, 1.0180861762150282,num_subjects))
rew_change=list(normalrnd(0.5043001380934354, 1.1065215386632883,num_subjects))

# lrf_mild=list(betarnd(2,5,num_subjects))
lrf_each=list(betarnd(3,5,num_subjects))

lr_cfr=list(betarnd(0.18712683635873772, 1.2398401783318014,num_subjects))
lr_cfp=list(betarnd(0.40,1.2,num_subjects))

lrf[0]=np.zeros((num_subjects))+0.1 # learnabout safe feature
# lrf[1]=lrf_mild # learn about mild feature (small reward)
lrf[1]=np.zeros((num_subjects))+0.1 # learn about predator feature

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
stick=list(normalrnd(0.10548073446686312, 0.5444583943396483,num_subjects))

# mf_betas=np.zeros((num_subjects,1))
weights=list(betarnd(7,3,num_subjects))
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

all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick,safe_first)
# all_data4,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf22,lrv2,betas2,mf_betas2,bandits,reward_function_timeseries2,pred_change2,rew_change2,lr_cfr2,lr_cfp2,rfi_rews2,rfi_preds2,stick2,safe_first1)
# all_data3=all_data3+all_data4
# In[10]:
import pandas as pd
from itertools import repeat,starmap
import numpy as np
import time
import scipy
from numpy.random import beta,gamma,chisquare,normal,poisson,uniform,logistic,multinomial,binomial



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

def feature_learner_lik_action_perseveration(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2

	beta_last=samples[0][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))

	lik=np.zeros((sample_size,))

	
	
	#account for data missing
	
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
		reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	for index in range(len(choice)):

		q_integrated=beta_last*last_choice
		
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
	

		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		
	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_no_cf(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2

	beta_last=samples[3][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))

	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))

	lr_features=samples[0][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[1][rng_samples]
	betas_pred=samples[2][rng_samples]


	#account for data missing
	
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
		reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	for index in range(len(choice)):
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)
		q_integrated=q_sr+beta_last*last_choice
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q

	

		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)

		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_no_cflr(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2

	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))

	beta_last=samples[3][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))

	lr_features=samples[0][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[1][rng_samples]
	betas_pred=samples[2][rng_samples]


	#account for data missing
	
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
		reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	for index in range(len(choice)):
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)
		q_integrated=q_sr+beta_last*last_choice
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q

	

		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)

		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		
		
	
	return lik

def feature_learner_lik_beta_counterfactual_two_empirical(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2

	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))

	lr_cf=samples[3][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	lr_features=samples[0][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[1][rng_samples]
	betas_pred=samples[2][rng_samples]

	beta_last=samples[4][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))



	#account for data missing
	
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
		reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	for index in range(len(choice)):
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)
		q_integrated=q_sr+beta_last*last_choice
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
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
		
		
	
	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_change(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	pred_changer=samples[3][rng_samples]
	rew_changer=samples[4][rng_samples]
	lr_cf=samples[5][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	lr_features=samples[0][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[1][rng_samples]
	betas_pred=samples[2][rng_samples]

	beta_last=samples[6][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))
	# mf_betas=samples[0][rng_samples]
	# mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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
		reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

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
		q_integrated=q_sr+beta_last*last_choice
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		other_decision=abs((int(choice[index])-1)-1)

		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		
		# q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_change_MF(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)

	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))

	#account for data missing
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
		reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

	elif reverse=='yes':
		reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
		outcome=data.reward_received
		safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
		predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

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
		q_integrated=q_sr+(q_values*mf_betas)+beta_last*last_choice
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
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
		
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik







def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive(samples,data,rng_samples):
	from scipy.special import logsumexp
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
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]


	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_GP
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

	return lik
def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_RFIchange(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	rew_rfi_changer=samples[10][rng_samples]
	pred_rfi_changer=samples[11][rng_samples]
	
	beta_last=samples[12][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)-rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)-pred_rfi_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)+rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)+pred_rfi_changer.reshape(sample_size,1)),axis=1)
		else:
			if index<80-cut_beginning:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)-rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)-pred_rfi_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)+rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)+pred_rfi_changer.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		
		q_integrated=q_sr+(q_values*mf_betas)+q_GP +(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik
def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_RFIchange_nochangeMB(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[6][rng_samples]
	rfi_pred=samples[7][rng_samples]
	rew_rfi_changer=samples[8][rng_samples]
	pred_rfi_changer=samples[9][rng_samples]
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	

	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)-rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)-pred_rfi_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)+rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)+pred_rfi_changer.reshape(sample_size,1)),axis=1)
		else:
			if index<80-cut_beginning:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)-rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)-pred_rfi_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
				rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1)+rew_rfi_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)+pred_rfi_changer.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		
		q_integrated=q_sr+(q_values*mf_betas)+q_GP +(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik




def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noMB(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[3][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	lr_val=samples[2][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	rfi_rew=samples[4][rng_samples]
	rfi_pred=samples[5][rng_samples]
	beta_last=samples[6][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	pred_changer=samples[7][rng_samples]
	rew_changer=samples[8][rng_samples]

	#account for data missing
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
				all_betas=np.concatenate((rfi_rew.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1-pred_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((rfi_rew.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1+pred_changer.reshape(sample_size,1)),axis=1)
		else:
			if index<80-cut_beginning:
				all_betas=np.concatenate((rfi_rew.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1-pred_changer.reshape(sample_size,1)),axis=1)
			else:
				all_betas=np.concatenate((rfi_rew.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1+pred_changer.reshape(sample_size,1)),axis=1)
		
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		
		q_rfi1=np.sum((feature_matrix1*all_betas),axis=1) 
		q_rfi2=np.sum((feature_matrix2*all_betas),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)


	return lik


def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_mfCF_nogp(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)

		current_reward_function=reward_function_timeseries[index]
		if -1 in current_reward_function:
			rf='pun'
		else:
			rf='rwd'
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q

		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		if rf=='pun':
			if predators[index][other_decision]==1:
				pe_cf=-1-q_values[:,other_decision]
			else:
				pe_cf=0-q_values[:,other_decision]
		elif rf=='rwd':
			if safes[index][other_decision]==1:
				pe_cf=1-q_values[:,other_decision]
			else:
				pe_cf=0-q_values[:,other_decision]
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)
		q_values[:,other_decision]=q_values[:,int(choice[index])-1]*(lr_val*pe_cf)

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_mfCF(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)

		current_reward_function=reward_function_timeseries[index]
		if -1 in current_reward_function:
			rf='pun'
		else:
			rf='rwd'
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)


		q_integrated=q_sr+(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q

		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		if rf=='pun':
			if predators[index][other_decision]==1:
				pe_cf=-1-q_values[:,other_decision]
			else:
				pe_cf=0-q_values[:,other_decision]
		elif rf=='rwd':
			if safes[index][other_decision]==1:
				pe_cf=1-q_values[:,other_decision]
			else:
				pe_cf=0-q_values[:,other_decision]
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)
		q_values[:,other_decision]=q_values[:,int(choice[index])-1]*(lr_val*pe_cf)

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)


	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_ignoreRF(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)


	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
	errors=data[data['chosen_state']=='SLOW']
	cut_beginning=0
	for i in errors.level_0:
		if i<81:
			cut_beginning+=1
	data = data[data['chosen_state']!='SLOW']
	data.dropna(subset=['chosen_state'], inplace=True)
	data=data.reset_index(drop=True)
	if data.safe_first[0]==True:
		safe_first=1
		
	else:
		safe_first=2
		
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

		if safe_first==1:
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
		current_reward_function_reverse=-1*np.flip(current_reward_function)
		if -1 in current_reward_function:
			current_bias=rfi_rew
		else:
			current_bias=rfi_pred
	

		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		current_weights_reverse=all_betas*current_reward_function_reverse
		q_sr1r=np.sum((feature_matrix1*current_weights_reverse),axis=1) 
		q_sr2r=np.sum((feature_matrix2*current_weights_reverse),axis=1)
		q_sr_r=np.concatenate((q_sr1r.reshape(sample_size,1),q_sr2r.reshape(sample_size,1)),axis=1)

		

		q_integrated=q_sr+(q_values*mf_betas)+(last_choice*beta_last)
		q_integrated_r=q_sr_r+(q_values*mf_betas)+(last_choice*beta_last)

		lik = lik + ((q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1))*(1-current_bias)+(q_integrated_r[:,int(choice[index])-1]-logsumexp(q_integrated_r,axis=1))*(current_bias))
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik


def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noMF(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[5][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[3][rng_samples]
	rew_changer=samples[4][rng_samples]

	lr_features=samples[0][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[1][rng_samples]
	betas_pred=samples[2][rng_samples]
	rfi_rew=samples[6][rng_samples]
	rfi_pred=samples[7][rng_samples]
	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)

	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)

	#account for data missing
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

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()


	return lik


def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noRR(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[5][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	# pred_changer=samples[5][rng_samples]
	# rew_changer=samples[6][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	rfi_rew=samples[6][rng_samples]
	rfi_pred=samples[7][rng_samples]
	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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
		# if sf==1:
		# 	if index>=80-cut_beginning:
		# 		all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
		# 	else:
		# 		all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		# else:
		# 	if index<80-cut_beginning:
		# 		all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
		# 	else:
		# 		all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
			
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik


def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noGP(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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


	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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


		q_integrated=q_sr+(q_values*mf_betas)+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik





def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_rfhistory(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	reward_rf_counter=np.zeros((sample_size,1))

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	pred_changer=pred_changer.reshape(sample_size,1)

	rew_changer=samples[6][rng_samples]
	rew_changer=rew_changer.reshape(sample_size,1)
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	rf_lr=samples[11][rng_samples]
	rf_lr=rf_lr.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		rew_ch=rew_changer*reward_rf_counter
		pred_ch=pred_changer*reward_rf_counter
	
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_ch,betas_pred.reshape(sample_size,1)+pred_ch),axis=1)
		
		current_reward_function=reward_function_timeseries[index]


		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		if 1 in current_reward_function:
			reward_rf_counter=reward_rf_counter*(1-rf_lr)+rf_lr
		else:
			reward_rf_counter=reward_rf_counter*(1-rf_lr)-rf_lr

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_MFRF_two_empirical_2lr_rfinsensitive_sticky(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	qmf_rew=np.zeros((sample_size,num_choices))
	qmf_pred=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))

	pred_changer=samples[3][rng_samples]
	rew_changer=samples[4][rng_samples]
	lr_features=samples[0][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[1][rng_samples]
	betas_pred=samples[2][rng_samples]
	rfi_rew=samples[5][rng_samples]
	rfi_pred=samples[6][rng_samples]
	
	lr_mf=samples[8][rng_samples]
	lr_mf=lr_mf.reshape(sample_size,1)
	lr_cf=samples[9][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	beta_last=samples[7][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)

	#account for data missing
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
				rew_betas=betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1)
				pred_betas=betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)
			else:
				rew_betas=betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1)
				pred_betas=betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)
		else:
			if index<80-cut_beginning:
				rew_betas=betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1)
				pred_betas=betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)
			else:
				rew_betas=betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1)
				pred_betas=betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)


		current_reward_function=reward_function_timeseries[index]
		if 1 in current_reward_function:
			q_mf=qmf_rew*rew_betas
		else:
			q_mf=qmf_pred*pred_betas 

		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)

		# current_reward_function=reward_function_timeseries[index]
		# current_weights=all_betas*current_reward_function
		# feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		# q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		# feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		# q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		# q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		
		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_mf+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)

		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1
		other_decision=abs((int(choice[index])-1)-1)


		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

		if 1 in current_reward_function:
			pe = outcome[index]-qmf_rew[:,int(choice[index])-1]
			qmf_rew[:,current_decision]=((qmf_rew[:,current_decision].reshape(sample_size,1)+(lr_mf*pe.reshape(sample_size,1)))).flatten()
		else:
			pe = outcome[index]-qmf_pred[:,int(choice[index])-1]
			qmf_pred[:,current_decision]=((qmf_pred[:,current_decision].reshape(sample_size,1)+(lr_mf*pe.reshape(sample_size,1)))).flatten()


	return lik





def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitiveMF_sticky(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	q_valuesS=np.zeros((sample_size,num_choices))
	q_valuesP=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[8][rng_samples]
	rfi_rew=rfi_rew.reshape(sample_size,1)
	rfi_pred=samples[9][rng_samples]
	rfi_pred=rfi_pred.reshape(sample_size,1)
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	lr_rfi=samples[11][rng_samples]
	# lr_rfi=lr_rfi.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		# rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+(q_valuesS*rfi_rew)+(last_choice*beta_last)+(q_valuesP*rfi_pred)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)

		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		pe = outcome[index]-q_values[:,int(choice[index])-1]
		value_safe=safes[index][current_decision]
		pe_safe_RFI=value_safe-q_valuesS[:,int(choice[index])-1]
		value_pred=predators[index][current_decision]*-1.0
		pe_pred_RFI=value_pred-q_valuesP[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)
		q_valuesS[:,int(choice[index])-1]=q_valuesS[:,int(choice[index])-1]+ (lr_rfi*pe_safe_RFI)
		q_valuesP[:,int(choice[index])-1]=q_valuesP[:,int(choice[index])-1]+ (lr_rfi*pe_pred_RFI)


	return lik






def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_mfContext(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi_rew=samples[8][rng_samples]
	rfi_pred=samples[9][rng_samples]
	beta_last=samples[10][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		if -1.0 in reward_function_timeseries[index]:
			if outcome[index]==0:
				reward=1
			else:
				reward=-1
		else:
			if outcome[index]==0:
				reward=-1
			else:
				reward=1
		pe = reward-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik

def guessing(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)
	lik=np.zeros((sample_size,))
	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))+0.5
	lr_cf=samples[0][rng_samples]

	for index in range(160):

		lik = lik + safe_q[:,0]-logsumexp(safe_q,axis=1)
	
	print(lik)

	return lik




def feature_learner_lik_beta_counterfactual_two_empirical_2lr_SINGLErfinsensitive_sticky(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	last_choice=np.zeros((sample_size,num_choices))
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
	rfi=samples[8][rng_samples]
	beta_last=samples[9][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	# order_changer=samples[9][rng_samples]
	# beta_rfinsensitive_r=samples[9][rng_samples]
	# betas_rfinsensitive_r=beta_rfinsensitive_r.reshape(sample_size,1)
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
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

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi.reshape(sample_size,1),rfi.reshape(sample_size,1)*-1.0),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_GP=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_GP+(last_choice*beta_last)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1

		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik






def feature_learner_lik_beta_counterfactual_two_empirical_justLR(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	safe_qr=np.zeros((sample_size,num_choices))
	pred_qr=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[5][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	# pred_changer=samples[6][rng_samples]
	# rew_changer=samples[7][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_safe=samples[1][rng_samples]
	lr_safe=lr_safe.reshape(sample_size,1)
	lr_pred=samples[2][rng_samples]
	lr_pred=lr_pred.reshape(sample_size,1)
	betas=samples[3][rng_samples]
	# betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing
	rfi_rew=samples[6][rng_samples]
	rfi_rew=rfi_rew.reshape(sample_size,1)
	rfi_pred=samples[7][rng_samples]
	rfi_pred=rfi_pred.reshape(sample_size,1)
	last_choice=np.zeros((sample_size,num_choices))
	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	rfi_betas=samples[9][rng_samples]
	rfi_betas=rfi_betas.reshape(sample_size,1)
	# lr_cf_gp=samples[12][rng_samples]
	# lr_cf_gp=lr_cf.reshape(sample_size,1)

	#account for data missing
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

		# if sf==1:
		# 	if index>=80-cut_beginning:
		# 		safe_lr=lr_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1)
		# 		pred_lr=lr_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)
		# 	else:
		# 		safe_lr=lr_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1)
		# 		pred_lr=lr_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)
		# else:
		# 	if index<80-cut_beginning:
		# 		safe_lr=lr_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1)
		# 		pred_lr=lr_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)
		# 	else:
		# 		safe_lr=lr_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1)
		# 		pred_lr=lr_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)

		all_betas=np.concatenate((betas.reshape(sample_size,1),betas.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		feature_matrix1=np.concatenate((safe_qr[:,0].reshape(sample_size,1),pred_qr[:,0].reshape(sample_size,1)),axis=1)
		feature_matrix2=np.concatenate((safe_qr[:,0].reshape(sample_size,1),pred_qr[:,0].reshape(sample_size,1)),axis=1)
		rf_insensitive_weights=np.concatenate((rfi_betas.reshape(sample_size,1),rfi_betas.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi+(last_choice*beta_last)
		
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q

		pe_safeqr=safes[index]-safe_qr
		pe_predqr=predators[index]-pred_qr
		
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		last_choice=np.zeros((sample_size,num_choices))
		last_choice[:,int(choice[index])-1]=1


		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_safe*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_pred*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		safe_qr[:,current_decision]=((safe_qr[:,current_decision].reshape(sample_size,1)+(rfi_rew*pe_safeqr[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_qr[:,other_decision]=((safe_qr[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safeqr[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_qr[:,current_decision]=((pred_qr[:,current_decision].reshape(sample_size,1)+(rfi_pred*pe_predqr[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_qr[:,other_decision]=((pred_qr[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_predqr[:,other_decision].reshape(sample_size,1)))).flatten()

		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

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
	# print('lik vector cleaned: {}'.format(likelihood_vector_cleaned))
	good_samples=len(likelihood_vector_cleaned) 

	weights=np.exp(likelihood_vector_cleaned) 
	# print('weights: {}'.format(likelihood_vector_cleaned))

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
			hyperparameters=beta.fit(parameter_full_sample,loc=0,scale=1)
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



def process_subject(subject,parameter_info,all_data,lik_func,parameter_sample_size):
		data=all_data[subject]
		data=data.reset_index()
		parameter_names=[x[0] for x in parameter_info]
		samples_a=sample_group_distributions_parallel(parameter_info,parameter_sample_size)

		rng=np.arange(0,parameter_sample_size,1)
		
		# compute likelihood
		likelihood=lik_func(samples_a,data,rng)
		likelihood=np.array(likelihood)

		#get cleaned likelihood vector
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







#################################################### GET SUBJECT DATA ####################################################################

import pickle
number_subjects=num_subjects
name_1='feature_learner_2states'
# number_subjects=1413

os.chdir('../data_and_main_analyses')
#load in empirical data
subj_IDs=[pd.read_csv(x) for x in os.listdir(os.curdir) if x.startswith('sub_3')]
sub_ID_names=[x.subjectID[10] for x in subj_IDs]


all_models=[]



number_subjects=192
# os.chdir('../../python/Feature_learning_anxiety')

############################################################# BUILD MODELS TO BE FIT #########################################################

parameter_sample_size=100000




# name_1='ap_only'
# group_parameters_info_m={	'ap':['norm',[0,1]]
						
# 							}
					 
# likelihood='feature_learner_lik_action_perseveration'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(ap_only)


# name_1='mb_ap_nocf'
# group_parameters_info_m={	'lr_f':['beta',[1,1]],
# 							'pred_s':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'ap':['norm',[0,1]]
						
# 							}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_no_cf'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_ap_nocf)


# name_1='mb_ap_nocflr'
# group_parameters_info_m={	'lr_f':['beta',[1,1]],
# 							'pred_s':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'ap':['norm',[0,1]]
						
# 							}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_no_cflr'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_ap_nocflr)


# name_1='mb_ap'
# group_parameters_info_m={	'lr_f':['beta',[1,1]],
# 							'pred_s':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'lr_cf':['beta',[1,1]],
# 							'ap':['norm',[0,1]]
						
# 							}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_ap)




# name_1='mb_rr_ap'
# group_parameters_info_m={	'lr_f':['beta',[1,1]],
# 							'pred_s':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'rew_ch':['norm',[0, 1]],
# 							'pred_ch':['norm',[0, 1]],
# 							'lr_cf':['beta',[1,1]],
# 							'ap':['norm',[0,1]]
						
# 							}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_change'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_ap)



# name_1='mb_rr_ap_mf'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 							'lr_f':['beta',[1,1]],
# 							'pred_s':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'lr_v':['beta',[1,1]],
# 							'rew_ch':['norm',[0, 1]],
# 							'pred_ch':['norm',[0, 1]],
# 							'lr_cf':['beta',[1,1]],
# 							'ap':['norm',[0,1]]
						
# 							}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_change_MF'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_ap_mf)


# name_1='mb_rfhistory_mf_gp_ap'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'st':['norm',[0, 1]],
# 						'lr_rf':['beta',[1,1]],
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_rfhistory'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rfhistory_mf_gp_ap)

# name_1='mb_rr_ap_mf_gp_Single'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'ap':['norm',[0,1]]
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_SINGLErfinsensitive_sticky'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_ap_mf_gp_Single)



# name_1='mb_rr_mf'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
						
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_mf)


# name_1='mb_rr_mf_gprr_ap'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'rew_chrfi':['norm',[0, 1]],
# 						'pred_chrfi':['norm',[0, 1]],
# 						'st':['norm',[0, 1]],
						
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_RFIchange'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_mf_gprr_ap)

# name_1='mb_mf_gprr_ap'
# group_parameters_info_m={'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
					
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'rew_chrfi':['norm',[0, 1]],
# 						'pred_chrfi':['norm',[0, 1]],
# 						'st':['norm',[0, 1]],
					
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_RFIchange_nochangeMB'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_mf_gprr_ap)

# name_1='gp_mf_ap'
# group_parameters_info_m={'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'lr_v':['beta',[1,1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'st':['norm',[0, 1]],
# 						'r_ch':['norm',[0, 1]],
# 						'p_ch':['norm',[0, 1]]
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noMB'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(gp_mf_ap)

# name_1='mbrr_gp_ap'
# group_parameters_info_m={
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
						
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'st':['norm',[0, 1]]
						
						
# 						}


					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noMF'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mbrr_gp_ap)

# name_1='mb_gp_mf_ap'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
						
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'st':['norm',[0, 1]]
						
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noRR'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_gp_mf_ap)

# name_1='mb_rr_mf_ap'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'st':['norm',[0, 1]]
						
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_noGP'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_mf_ap)


# name_1='MFRF'
# group_parameters_info_m={	
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'st':['norm',[0, 1]],
# 						'lr_mf':['beta',[1,1]],
# 						'lr_cf':['beta',[1,1]]
						
# 						}
					 
# likelihood='feature_learner_lik_beta_MFRF_two_empirical_2lr_rfinsensitive_sticky'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(MFRF)

# name_1='mb_rr_ap_m3'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'ap':['norm',[0,1]],
# 						'lr_rfi':['beta',[1,1]],
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitiveMF_sticky'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_ap_m3)


# name_1='MB_GP_BOTH_LR'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
						
# 						'pred_s':['beta',[1, 1]],
# 						'pred_b':['beta',[1, 1]],
# 						'betas':['gamma',[1,1]],
# 						'lr_v':['beta',[1,1]],
# 						'lr_cf':['beta',[1,1]],
						
# 						'gp_rew':['beta',[1, 1]],
# 						'gp_prew':['beta',[1, 1]],
# 						'ap':['norm',[0,1]],
# 						'beta_rfi':['gamma',[1,1]],
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_justLR'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(MB_GP_BOTH_LR)


# name_1='mb_rr_mf_gp_ap'
# group_parameters_info_m={	'mf':['gamma',[1, 1]],
# 						'lr_f':['beta',[1,1]],
# 						'pred_s':['gamma',[1, 1]],
# 						'pred_b':['gamma',[1, 1]],
# 						'lr_v':['beta',[1,1]],
# 						'rew_ch':['norm',[0, 1]],
# 						'pred_ch':['norm',[0, 1]],
# 						'lr_cf':['beta',[1,1]],
# 						'gp_rew':['gamma',[1, 1]],
# 						'gp_prew':['gamma',[1, 1]],
# 						'st':['norm',[0, 1]],
						
						
# 						}
					 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_RFIchange'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(mb_rr_mf_gp_ap)

name_1='mb_rr_mf_distraction_ap'
group_parameters_info_m={'mf':['gamma',[1, 1]],
						'lr_f':['beta',[1,1]],
						'pred_s':['gamma',[1, 1]],
						'pred_b':['gamma',[1, 1]],
						'lr_v':['beta',[1,1]],
						'rew_ch':['norm',[0, 1]],
						'pred_ch':['norm',[0, 1]],
						'lr_cf':['beta',[1,1]],
						'gp_rew':['beta',[1, 1]],
						'gp_prew':['beta',[1, 1]],
						'st':['norm',[0, 1]],
						
						
						}
					 
likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_ignoreRF'
build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
all_models.append(mb_rr_mf_distraction_ap)

#define cores:
cores_subs=14

############################################################### FIT MODELS #############################################################

import concurrent.futures
from multiprocessing.dummy import Pool
import multiprocessing
from itertools import repeat,starmap
import numpy as np
import time




#current_dataset=all_data_input #make sure dataset is correct before running model fitting!!
current_dataset=subj_IDs
print(len(subj_IDs))
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
			repeat(lik_func),repeat(parameter_sample_size))
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
		print(total_evidence)
		new_bic = -2.0*float(total_evidence) + Nparams*np.log(Ndatapoints) # Bayesian Information Criterion
		improvement = old_bic - new_bic # compute improvement of fit
		
		#only retain evidence and BIC if they improve
		if improvement > 0:
			model.model_evidence_group=total_evidence
			model.bic=new_bic
		
		# read out latest iteration
		print('{}- iBIC old:{}, new: {}\n'.format(model.name, old_bic, new_bic))
	models_BICs_record.append(model.bic)
	print(models_BICs_record)

print('current_hyperparms:')
print(parameter_info)
print('final param info:')
print(models_BICs_record)
print('')



################################################### PRINT OUT FITTED PARAMETERS #########################################################################

beta_s=[]
beta_m=[]
beta_p=[]
pre_change=[]
re_change=[]
lr_cf=[]
lrcf_pred=[]
zero=[]
one=[]
two=[]
lr_val=[]
lr_neg=[]
lr_pos=[]
beta_rfis_p=[]
beta_rfis_r=[]
beta_rfis_pc=[]
beta_rfis_rc=[]
beta_st=[]
beta_st2=[]
rf_lr=[]
psens_rfis=[]

for subject in range(number_subjects):
	zero.append(all_results[subject][0])
	one.append(all_results[subject][1])
	two.append(all_results[subject][2][1]) 
	lr_neg.append(all_results[subject][3][1]) 
	beta_s.append(all_results[subject][4][1]) 
	beta_p.append(all_results[subject][5][1]) 
	lr_val.append(all_results[subject][6][1]) 
	pre_change.append(all_results[subject][7][1]) 
	re_change.append(all_results[subject][8][1])
	lr_cf.append(all_results[subject][9][1]) 
	beta_rfis_r.append(all_results[subject][10][1])  
	beta_rfis_p.append(all_results[subject][11][1])
	beta_st.append(all_results[subject][12][1])

lrs=np.asarray(lr_neg)
# np.save('lr_features',lrs)
mbs=np.asarray(beta_s)
# np.save('mb_rew_betas',mbs)
mbp=np.asarray(beta_p)
# np.save('mb_pun_betas',mbp)
lrv=np.asarray(lr_val)
# np.save('lrvs',lrv)
pch=np.asarray(pre_change)
# np.save('change_pun',pch)
rch=np.asarray(re_change)
# np.save('change_rew',rch)
lrcf=np.asarray(lr_cf)
# np.save('lrcfs',lrcf)
gpr=np.asarray(beta_rfis_r)
# np.save('gprew_betas',gpr)
gpp=np.asarray(beta_rfis_p)
# np.save('gppun_betas',gpp)
aps=np.asarray(beta_st)
# np.save('ap_betas',aps)
mfb=np.asarray(two)
# np.save('mf_betas',mfb)

print('mf beta: mean{}, sd: {}'.format(np.mean(two),np.std(two))) #mfb
print(two)
print('')


print('lr cf: mean{}, sd: {}'.format(np.mean(lr_cf),np.std(lr_cf))) #lr val
print(lr_cf)
print('')

print('feature LR FEATURE: mean{}, sd: {}'.format(np.mean(lr_neg),np.std(lr_neg)))
print(lr_neg)
print('')

print('feature MB safe: mean{}, sd: {}'.format(np.mean(beta_s),np.std(beta_s))) #lr feature
print(beta_s)
print('')

print('feature MB pred: mean{}, sd: {}'.format(np.mean(beta_p),np.std(beta_p))) #beta s
print(beta_p)
print('')

print('rew change: mean{}, sd: {}'.format(np.mean(re_change),np.std(re_change))) # pre change
print(re_change)
print('')

print('pre change: mean{}, sd: {}'.format(np.mean(pre_change),np.std(pre_change))) #lr cf
print(pre_change)
print('')

print('feature beta RFI rew: mean{}, sd: {}'.format(np.mean(beta_rfis_r),np.std(beta_rfis_r))) #re change
print(beta_rfis_r)
print('')

print('feature beta RFI pred: mean{}, sd: {}'.format(np.mean(beta_rfis_p),np.std(beta_rfis_p))) #rfi r
print(beta_rfis_p)
print('')


print('feature betas sticky: mean{}, sd: {}'.format(np.mean(beta_st),np.std(beta_st)))
print(beta_st)
print('')

print('feature LR VALUE: mean{}, sd: {}'.format(np.mean(lr_val),np.std(lr_val))) #beta p
print(lr_val)
print('')


############################################### ONLY FOR SIMULATED DATA: PARAMETER RECOVERY ##########################################################
import matplotlib.pyplot as plt

# # Generate a large random dataset
# d = pd.DataFrame()
# d['lr value']=lrv
# d['MF']=mf_betas
# d['lr feature']=lrf_safe
# d['MB reward']=beta_safe
# d['MB punish']=beta_pred
# d['Ch reward']=rew_change
# d['Ch punish']=pred_change
# d['lr cf']=lr_cfr
# d['GI reward']=rfi_rews
# d['GI punish']=rfi_preds
# d['Perseveration']=stick
# d['S lr value ']=lr_val
# d['S MF']=two
# d['S lrfeature']=lr_neg
# d['S MB reward']=beta_s
# d['S MB punish']=beta_p
# d['S Ch reward']=re_change
# d['S Ch punish']=pre_change
# d['S lr cf']=lr_cf
# d['S GI reward']=beta_rfis_r
# d['S GI punish']=beta_rfis_p
# d['S Perseveration']=beta_st


# # Compute the correlation matrix
# corr = d.corr()
# # corr.to_csv('winning_corr_matrix_fixed2.csv')
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, cmap=cmap, center=0,
#             square=True, linewidths=.5)

# plt.show()