#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Bayesian model fitting via iterative sampling
# 
# This algorithm iteratively computes model evidence via a sampling procedure. Each iteration of the loop a posterior sample for each subject is derived by weighting the distribution of parameters by their likelihood. These samples at the subject level are concatenated, and then hyperparameters are fit to these distributions. Model evidence at the group level is a sum of subject-level evidence (given that is in log space). Once model evidence (defined by iBIC) does not improve, the iterations terminate. 
# 
# For each subject
# 
# - Sample from a prior distribution over $\forall i$ paramters $\theta_{i}$
# 
# - Gennerate likelihoods $p(data|\theta_i)$
# 
# - Compute posterior distribution over parameters at individual level by $p(\theta)\cdot p(data|\theta)$
# 
# For group:
# 
# - Fit new hyperparameters to the full sample (concatenate all individual-level posteriors) then go back to (1) unless iBIC hasn't improved

# # Function to generate random walks over features

# In[1]:

import os



def generate_bandits(starting_bandits,num_trials,num_momentum_switches,random_walk,momentum_mean,momentum_variance):
	import numpy as np
	from random import shuffle
	num_bandits=len(starting_bandits)
	shuffle(starting_bandits)
	value_scale=1

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

def simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st):
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
				if trial<=80:
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

				q_irf_weights=np.array((rfi_rew,-1*rfi_pred))
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
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds

def simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky_ignoreRF(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st):
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
				if trial<=80:
					beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))
					
				else:
					beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
				  


				# beta_weights_SR=np.array((beta_safe,beta_pred))

						
				# participant is told which predator they're facing
				if -1 in reward_function_time_series[trial]:
					coin_flip=np.random.binomial(1, 0.8, 1)[0]
					if coin_flip==1:
						current_reward_function=reward_function_time_series[trial]
					else:
						current_reward_function=-1*np.flip(reward_function_time_series[trial])

				else:
					coin_flip=np.random.binomial(1, 0.6, 1)[0]
					if coin_flip==1:
						current_reward_function=reward_function_time_series[trial]
					else:
						current_reward_function=-1*np.flip(reward_function_time_series[trial])




				#weighting is a combo of beta weights and reward function
				full_weights=beta_weights_SR*current_reward_function
				#derive SR q-values by multiplying feature vectors by reward function
				q_sr=np.matmul(feature_matrix,full_weights)
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values))+((last_beta)*(last_choice))
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
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds



def simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky_psens(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st,psens):
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
			psensitivity=psens[subject]
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
				if -1 in current_reward_function:
					current_reward_function=current_reward_function*psensitivity

				#weighting is a combo of beta weights and reward function
				full_weights=beta_weights_SR*current_reward_function
				#derive SR q-values by multiplying feature vectors by reward function
				q_sr=np.matmul(feature_matrix,full_weights)

				q_irf_weights=np.array((rfi_rew,-1*rfi_pred*psensitivity))
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
			exec('dataset_sub_{}_game_{}["pred_o"]=preds'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["safe_o"]=safes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
			exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
			exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

			
		
	return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds


def simulate_one_step_featurelearner_counterfactual_2states_2iRF_gradsticky(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st,lr_sticks):
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
			lr_stick=lr_sticks[subject]
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

				q_irf_weights=np.array((rfi_rew,-1*rfi_pred))
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
				last_choice[current_decision]=(last_choice[current_decision]*(1-lr_stick))+lr_stick

				other_decision=abs(current_choice-1)
				last_choice[other_decision]=(last_choice[other_decision]*(1-lr_stick))
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
##### GENERATE SYNTHETIC DATA

# #### import pandas as pd
import numpy as np
from numpy.random import beta as betarnd
from numpy.random import gamma as gammarnd
from numpy.random import normal as normalrnd
from random import sample
from random import shuffle
import pandas as pd

# [['beta_mfRFI', 'gamma', [1.2490709436845728, 0.5765035173332131]], 
# ['lr_neg', 'beta', [0.36721628392270533, 1.3164765310611757]], 
# ['pred_s', 'gamma', [0.6972989020743176, 5.586658152923879]],
#  ['pred_b', 'gamma', [0.9287736384634673, 2.0707066188048318]], 
#  ['lr_val', 'beta', [0.7296034278976317, 0.6172145553720277]], 
#  ['pre_change', 'norm', [-0.17894234198920192, 1.025507372903099]], 
#  ['re_change', 'norm', [0.39464442406144595, 0.8931352415845109]], 
#  ['lr_cf', 'beta', [0.21028678616344307, 1.124880118297063]], 
#  ['w_rfi_rew', 'gamma', [0.6581874232132539, 0.9304058132594626]], 
#  ['w_rfi_pred', 'gamma', [0.9235182855547278, 1.0358933472043663]], 
#  ['stick1', 'norm', [0.1294211722464285, 0.5602493849326147]], 
#  ['pensitivity', 'gamma', [1.1632070788188429, 1.0031407199969151]]]



#  [['beta_mf', 'gamma', [1.1093447925340514, 0.548557708547694]],
#   ['lr_neg', 'beta', [0.3238920704407903, 1.3271625781269714]], 
#   ['pred_s', 'gamma', [0.6025122100072803, 6.731039259186533]], 
#   ['pred_b', 'gamma', [0.4529044061222983, 4.2339741652612215]],
#   ['lr_val', 'beta', [0.821798751893812, 0.49956250235199756]], 
#   ['pre_change', 'norm', [-0.14914530448854793, 1.0180861762150282]],
#    ['re_change', 'norm', [0.5043001380934354, 1.1065215386632883]],
#     ['lr_cf', 'beta', [0.18712683635873772, 1.2398401783318014]],
#      ['beta_rfi_rew', 'gamma', [0.4275512172820504, 1.6876320240266436]],
#       ['beta_rfi_pred', 'gamma', [0.6019802760339684, 1.4308407307549147]], 
#       ['stick', 'norm', [0.10548073446686312, 0.5444583943396483]]]
# #assumes you've generated bandits variable with the block above
df=pd.DataFrame()
num_subjects=80
num_trials=160
lrv=list(betarnd(0.7296034278976317, 0.6172145553720277,num_subjects))
lrf=np.zeros((2,num_subjects))
lrf_safe=list(betarnd(0.36721628392270533, 1.3164765310611757,num_subjects))
lrf_pos=list(betarnd(0.6,2.0,num_subjects))
lrf_neg=list(betarnd(0.36721628392270533, 1.3164765310611757,num_subjects))

pred_change=list(normalrnd(-0.17894234198920192, 1.025507372903099,num_subjects))
rew_change=list(normalrnd(0.39464442406144595, 0.8931352415845109,num_subjects))

# lrf_mild=list(betarnd(2,5,num_subjects))
lrf_each=list(betarnd(3,5,num_subjects))

lr_cfr=list(betarnd(0.21028678616344307, 1.124880118297063,num_subjects))
lr_cfp=list(betarnd(0.21028678616344307, 1.124880118297063,num_subjects))

lrf[0]=np.zeros((num_subjects))+0.1 # learnabout safe feature
# lrf[1]=lrf_mild # learn about mild feature (small reward)
lrf[1]=np.zeros((num_subjects))+0.1 # learn about predator feature

betas=np.zeros((2,num_subjects))
# beta_safe=list(gammarnd(0.75,6,num_subjects))
# beta_pred=list(gammarnd(0.62,3.71,num_subjects))
beta_safe=list(gammarnd(0.6025122100072803, 6.731039259186533,num_subjects))
beta_pred=list(gammarnd(0.6025122100072803, 6.731039259186533,num_subjects))
betas[0]=beta_safe
#np.zeros((num_subjects))+3.0 # learnabout safe feature
# betas[1]=beta_mild # learn about mild feature (small reward)
betas[1]=beta_pred
#np.zeros((num_subjects))+3.0 # learn about predator feature

#for beta model, there is only a single learning rate over features
lrf2=np.zeros((2,num_subjects))
lrf2[0]=lrf_safe # learnabout safe feature
lrf2[1]=lrf_safe # learnabout safe feature
# lrf2[2]=lrf_safe # learnabout safe feature


it=list(gammarnd(1,1,num_subjects))
mf_betas=list(gammarnd(1.2490709436845728, 0.5765035173332131,num_subjects))
# beta_irf=list(gammarnd(0.29,3.92,num_subjects))
rfi_rews=list(gammarnd(0.4275512172820504, 1.6876320240266436,num_subjects))
rfi_preds=list(gammarnd(0.6019802760339684, 1.4308407307549147,num_subjects))
psenss=list(gammarnd(1.1632070788188429, 1.0031407199969151,num_subjects))
stick=list(normalrnd(0.1294211722464285, 0.5602493849326147,num_subjects))
lr_sticks=list(betarnd(.62,.81,num_subjects))
# mf_betas=np.zeros((num_subjects,1))
weights=list(betarnd(7,3,num_subjects))
bandits=np.load('safe_firsta.npy')
rf2=np.load('rf_safe_80trials.npy')
# rf2=np.flip(rf2,1)

rf2=rf2/3.0
rf1=np.load('rf_pred_80trials.npy')
# rf1=np.flip(rf1,1)

rf1=rf1/3.0


reward_function_timeseries=np.concatenate((rf2,rf1),axis=0)

#CREATE SYNTHETIC DATA
all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky_ignoreRF(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)

# all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky_psens(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick,psenss)

# In[10]:
import pandas as pd
from itertools import repeat,starmap
import numpy as np
import time
import scipy
from numpy.random import beta,gamma,chisquare,normal,poisson,uniform,logistic,multinomial,binomial
# from numba import jit



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

# from numba import jit
# from numba import int32, float32    # import the types


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

def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_sticky_Psensitivity(samples,data,rng_samples):
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
	p_sens=samples[11][rng_samples]
	p_sens=p_sens.reshape(sample_size,1)
	
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)


	data=data.reset_index(drop=True)


	#grab relevant data
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
	

		current_reward_function=reward_function_timeseries[index]
		if -1 in current_reward_function:
			current_weights=all_betas*current_reward_function*p_sens
		else:
			current_weights=all_betas*current_reward_function
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)


		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1*p_sens),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred+(last_choice*beta_last)
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

def feature_learner_lik_beta_counterfactual_two_change_2iR_sticky(samples,data,rng_samples):

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
	
	for index in range(len(choice)):
		if index<=80:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)+rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)+pred_changer.reshape(sample_size,1)),axis=1)
		else:
			all_betas=np.concatenate((betas_safe.reshape(sample_size,1)-rew_changer.reshape(sample_size,1),betas_pred.reshape(sample_size,1)-pred_changer.reshape(sample_size,1)),axis=1)
		# all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function
	   

		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)

		
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)


		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1), axis=1)
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

def feature_learner_lik_beta_counterfactual_two_change_2iR_sticky_noGP(samples,data,rng_samples):

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
	lr_val=samples[4][rng_samples]
	lr_features=samples[1][rng_samples]
	lr_features=lr_features.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	beta_last=samples[8][rng_samples]
	beta_last=beta_last.reshape(sample_size,1)
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	predators=data.pred_o
	
	for index in range(len(choice)):
		if index<=80:
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

		q_integrated=q_sr+(q_values*mf_betas)+(last_choice*beta_last)
		
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

def feature_learner_lik_beta_counterfactual_two_change_2iR_gradualsticky(samples,data,rng_samples):

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
	lr_sticky=samples[11][rng_samples]
	lr_sticky=lr_sticky.reshape(sample_size,1)
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


		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1), axis=1)
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
		# last_choice=np.zeros((sample_size,num_choices))
		lc=last_choice[:,current_decision].reshape(sample_size,1)
		oc=last_choice[:,other_decision].reshape(sample_size,1)
		last_choice[:,current_decision]=(((1-lr_sticky)*lc)+lr_sticky).reshape(sample_size,)
		last_choice[:,other_decision]=((1-lr_sticky)*oc).reshape(sample_size,)

		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik


def feature_learner_lik_beta_counterfactual_two_peLR(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	
	lik=np.zeros((sample_size,))
	lr_cf=samples[7][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	# lr_cf_p=samples[9][rng_samples]
	# lr_cf_p=lr_cf_p.reshape(sample_size,1)
	pred_changer=samples[5][rng_samples]
	rew_changer=samples[6][rng_samples]
	lr_val=samples[4][rng_samples]
	lr_neg=samples[1][rng_samples]
	lr_neg=lr_neg.reshape(sample_size,1)
	lr_pos=samples[8][rng_samples]
	lr_pos=lr_pos.reshape(sample_size,1)
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
		if safes[index][current_decision]==0:
			safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_neg*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		elif safes[index][current_decision]==1:
			safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_pos*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()

		if predators[index][current_decision]==0:
			pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_neg*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		elif predators[index][current_decision]==1:
			pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_pos*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_2lr(samples,data,rng_samples):
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
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred
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

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		
		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred +(last_choice*beta_last)
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

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		
		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred +(last_choice*beta_last)
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

		# rf_insensitive_weights=betas_rfinsensitive_r*np.array((1.0,0))
		# q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		# q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		# q_rfi_rew=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		rf_insensitive_weights=np.concatenate((rfi_rew.reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred+(last_choice*beta_last)
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
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred+(last_choice*beta_last)
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

def feature_learner_lik_beta_counterfactual_two_empirical_weightedprobsMBrfi_sticky(samples,data,rng_samples):
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
		
		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		weighted_feature_probs1=feature_matrix1*all_betas
		q_sr1=np.sum((weighted_feature_probs1*current_reward_function),axis=1)  
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		weighted_feature_probs2=feature_matrix2*current_reward_function
		q_sr2=np.sum((weighted_feature_probs2*current_reward_function),axis=1)
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1)),axis=1)

		
		rfi_rf=np.array((1.0,-1.0))
		q_rfi1=np.sum((weighted_feature_probs1*rfi_rf),axis=1) 
		q_rfi2=np.sum((weighted_feature_probs2*rfi_rf),axis=1)
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred+(last_choice*beta_last)
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

def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitivepred_sticky(samples,data,rng_samples):
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
	rfi_pred=samples[8][rng_samples]
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

		rf_insensitive_weights=np.concatenate((np.zeros(sample_size).reshape(sample_size,1),rfi_pred.reshape(sample_size,1)*-1),axis=1)
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi_pred=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
		# +q_rfi_rew add in if want reward insensitive

		q_integrated=q_sr+(q_values*mf_betas)+q_rfi_pred+(last_choice*beta_last)
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


def feature_learner_lik_beta_counterfactual_two_empirical_2lr_rfinsensitive_noMF(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
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
	beta_rfinsensitive=samples[0][rng_samples]
	betas_rfinsensitive=beta_rfinsensitive.reshape(sample_size,1)
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

		rf_insensitive_weights=betas_rfinsensitive*np.array((1.0,-1.0))
		q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
		q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
		q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+q_rfi
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_features*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		

	return lik
def feature_learner_lik_beta_counterfactual_two_empirical_peLR_cf(samples,data,rng_samples):
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
	lr_neg=samples[1][rng_samples]
	lr_neg=lr_neg.reshape(sample_size,1)
	lr_pos=samples[8][rng_samples]
	lr_pos=lr_pos.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
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
		q_integrated=q_sr+(q_values*mf_betas)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		if safes[index][current_decision]==0:
			safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_neg*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		if safes[index][current_decision]==1:
			safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_pos*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()

		if predators[index][current_decision]==0:
			pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_neg*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		if predators[index][current_decision]==0:
			pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_pos*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_peLR(samples,data,rng_samples):
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
	lr_neg=samples[1][rng_samples]
	lr_neg=lr_neg.reshape(sample_size,1)
	lr_pos=samples[7][rng_samples]
	lr_pos=lr_pos.reshape(sample_size,1)
	betas_safe=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]
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
		q_integrated=q_sr+(q_values*mf_betas)
		lik = lik + q_integrated[:,int(choice[index])-1]-logsumexp(q_integrated,axis=1)
		pe = outcome[index]-q_values[:,int(choice[index])-1]
		pe_safe=safes[index]-safe_q
		pe_pred=predators[index]-pred_q
		
		current_decision=int(choice[index])-1
		other_decision=abs((int(choice[index])-1)-1)
		if safes[index][current_decision]==0:
			safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_neg*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		elif safes[index][current_decision]==1:
			safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_pos*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()

		if predators[index][current_decision]==0:
			pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_neg*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		elif predators[index][current_decision]==0:
			pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_pos*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		
		if safes[index][other_decision]==0:
			safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_neg*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		elif safes[index][other_decision]==1:
			safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_pos*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()

		if predators[index][other_decision]==0:
			pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_neg*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		elif predators[index][other_decision]==0:
			pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_pos*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
	return lik

def feature_learner_lik_beta_counterfactual_two_empirical_justLR(samples,data,rng_samples):
	from scipy.special import logsumexp
	sample_size=len(rng_samples)

	num_choices=2
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	lik=np.zeros((sample_size,))
	lr_cf=samples[4][rng_samples]
	lr_cf=lr_cf.reshape(sample_size,1)
	# pred_changer=samples[5][rng_samples]
	# rew_changer=samples[6][rng_samples]
	lr_val=samples[3][rng_samples]
	lr_safe=samples[1][rng_samples]
	lr_safe=lr_safe.reshape(sample_size,1)
	lr_pred=samples[5][rng_samples]
	lr_pred=lr_pred.reshape(sample_size,1)
	betas=samples[2][rng_samples]
	# betas_pred=samples[3][rng_samples]
	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	#account for data missing

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
	
		all_betas=np.concatenate((betas.reshape(sample_size,1),betas.reshape(sample_size,1)),axis=1)
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
		safe_q[:,current_decision]=((safe_q[:,current_decision].reshape(sample_size,1)+(lr_safe*pe_safe[:,current_decision].reshape(sample_size,1)))).flatten()
		safe_q[:,other_decision]=((safe_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_safe[:,other_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,current_decision]=((pred_q[:,current_decision].reshape(sample_size,1)+(lr_pred*pe_pred[:,current_decision].reshape(sample_size,1)))).flatten()
		pred_q[:,other_decision]=((pred_q[:,other_decision].reshape(sample_size,1)+(lr_cf*pe_pred[:,other_decision].reshape(sample_size,1)))).flatten()
		
		q_values[:,int(choice[index])-1]=q_values[:,int(choice[index])-1]+ (lr_val*pe)

	return lik
def feature_learner_lik_beta_counterfactual_flr(samples,data,rng_samples):

	sample_size=len(rng_samples)

	num_choices=int(max(data.choices))
	q_values=np.zeros((sample_size,num_choices))
	safe_q=np.zeros((sample_size,num_choices))
	mild_q=np.zeros((sample_size,num_choices))
	pred_q=np.zeros((sample_size,num_choices))
	
	lik=np.zeros((sample_size,))
	lr_val=samples[4][rng_samples]
	lr_features=0.2
	betas_safe=samples[1][rng_samples]
	betas_mild=samples[2][rng_samples]
	betas_pred=samples[3][rng_samples]

	mf_betas=samples[0][rng_samples]
	mf_betas=mf_betas.reshape(sample_size,1)
	
	reward_function_timeseries=data.rf
	choice=data.choices
	outcome=data.outcomes
	safes=data.safe_o
	milds=data.mild_o
	predators=data.pred_o
	
	for index in range(len(choice)):
		all_betas=np.concatenate((betas_safe.reshape(sample_size,1),betas_mild.reshape(sample_size,1),betas_pred.reshape(sample_size,1)),axis=1)
		current_reward_function=reward_function_timeseries[index]
		current_weights=all_betas*current_reward_function


		feature_matrix1=np.concatenate((safe_q[:,0].reshape(sample_size,1),mild_q[:,0].reshape(sample_size,1),pred_q[:,0].reshape(sample_size,1)),axis=1)
		q_sr1=np.sum((feature_matrix1*current_weights),axis=1) 
		feature_matrix2=np.concatenate((safe_q[:,1].reshape(sample_size,1),mild_q[:,1].reshape(sample_size,1),pred_q[:,1].reshape(sample_size,1)),axis=1)
		q_sr2=np.sum((feature_matrix2*current_weights),axis=1)
		feature_matrix3=np.concatenate((safe_q[:,2].reshape(sample_size,1),mild_q[:,2].reshape(sample_size,1),pred_q[:,2].reshape(sample_size,1)),axis=1)
		q_sr3=np.sum((feature_matrix3*current_weights),axis=1) 
		
		q_sr=np.concatenate((q_sr1.reshape(sample_size,1),q_sr2.reshape(sample_size,1),q_sr3.reshape(sample_size,1)),axis=1)

		q_integrated=q_sr+(q_values*mf_betas)
		
		lik = lik + q_integrated[:,int(choice[index]-1)]-scipy.special.logsumexp(q_integrated,axis=1)

		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		pe_safe=safes[index]-safe_q
		pe_mild=milds[index]-mild_q
		pe_pred=predators[index]-pred_q

		safe_q=safe_q+(lr_features*pe_safe)
		mild_q=mild_q+(lr_features*pe_mild)
		pred_q=pred_q+(lr_features*pe_pred)
		
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)

	return lik
def std_RL_p(samples,data,rng_samples): #for parallel computing

	sample_size=len(rng_samples)
	
	num_choices=int(max(data.choices)+1)
	#initialize variables at zero
	q_values=np.zeros((sample_size,num_choices)) #initialize values for each sample, assumes 2 choices
	lik=np.zeros((sample_size,))
	
	#retrieve samples of parameters from model 
	lr_val=samples[1][rng_samples]
	inv_temp=samples[0][rng_samples]
	
	#retrieve relevant data
	choice=data.choices # data = pandas dataframe
	outcome=data.outcomes
	
	#calculate log likelihood by iterating over choice data
	for index in range(len(choice)):
		lik = lik + ((inv_temp*q_values[:,int(choice[index]-1)])-(scipy.special.logsumexp((q_values.transpose()*inv_temp).transpose(),axis=1)))
		pe = outcome[index]-q_values[:,int(choice[index]-1)]
		q_values[:,int(choice[index]-1)]=q_values[:,int(choice[index]-1)]+ (lr_val*pe)
	return lik
		  
def moodyRL(model,data,rng_samples):

	sample_size=model.sample_size
	
	num_choices=max(data.choices)+1
	#initialize variables at zero
	q_values=np.zeros((model.sample_size,num_choices)) #initialize values for each sample, assumes 2 choices
	lik=np.zeros((model.sample_size,))
	mood=np.zeros((model.sample_size,))
	
	#retrieve samples of parameters from model 
	lr_mood=model.lr_mood.samples
	lr_val=model.lr_val.samples
	inv_temp=model.inv_temp.samples
	mood_bias=model.mood_bias.samples
	
	#retrieve relevant data
	choice=data.choices # data = pandas dataframe
	outcome=data.outcomes
	
	#calculate log likelihood by iterating over choice data
	for index in range(len(choice)):
		x=inv_temp*q_values[:,choice[index]]
		lik = lik + ((inv_temp*q_values[:,choice[index]])-(scipy.special.logsumexp((q_values.transpose()*inv_temp).transpose(),axis=1)))
		uf=np.tanh(mood); #nonlinear transformation of mood by sigmoid
		perceived_r=outcome[index]+np.multiply(mood_bias,uf)
		pe = perceived_r-q_values[:,choice[index]]
		mood=mood*(1-lr_mood)+lr_mood*pe
		q_values[:,choice[index]]=q_values[:,choice[index]]+ (lr_val*pe)   
	return lik

def std_RL(model,data):

	sample_size=model.sample_size
	
	num_choices=int(max(data.choices)+1)
	#initialize variables at zero
	q_values=np.zeros((model.sample_size,num_choices)) #initialize values for each sample, assumes 2 choices
	lik=np.zeros((model.sample_size,))
	
	#retrieve samples of parameters from model 
	lr_val=model.lr_val.samples
	inv_temp=model.inv_temp.samples
	
	#retrieve relevant data
	choice=data.choices # data = pandas dataframe
	outcome=data.outcomes
	
	#calculate log likelihood by iterating over choice data
	for index in range(len(choice)):
		lik = lik + ((inv_temp*q_values[:,choice[index]])-(scipy.special.logsumexp((q_values.transpose()*inv_temp).transpose(),axis=1)))
		pe = outcome[index]-q_values[:,choice[index]]
		q_values[:,choice[index]]=q_values[:,choice[index]]+ (lr_val*pe)
	return lik

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
# # Build models

# In[11]:


all_models=[]

#define the number of subjects, parameters, and hyperparameters (i.e., parameters for group prior)

# # Complex Feature Learner -- Different learning rates for each feature
# name_1='feature_learner'
# number_subjects=num_subjects
# parameter_sample_size=96000
#                             #name of parameters and hyperpriors
# group_parameters_info_mood={'inv_temp':['gamma',[1,1]],'lr_features_safes':['beta',[1,1]],
#                             'lr_features_milds':['beta',[1,1]],'lr_features_preds':['beta',[1,1]],
#                             'weight':['beta',[1,1]],'lr_val':['beta',[1,1]]} 
# likelihood='feature_learner_lik'
# build_model(name_1,likelihood,group_parameters_info_mood,number_subjects,parameter_sample_size)
# all_models.append(feature_learner)


# Complex Feature Learner -- Different learning rates for each feature
	#(mf_betas,lr_features,beta-1-2-3,lrv,)

import pickle

# with open("dataframes_dangerEnv.bin","rb") as data:
#     all_data_input=pickle.load(data)



#### NUMBER SUBJECT #####

number_subjects=num_subjects
#number_subjects=40
# stop
# os.chdir('../../python/Feature_learning_anxiety')



parameter_sample_size=100000

# name_1='feature_learner_2states_justLR'
# group_parameters_info_m={'be_mf':['gamma',[1,1]],
#                             'lr_safe':['beta',[1,1]],
#                             'betaf':['gamma',[1,1]],
#                             'lr_val':['beta',[1,1]],
#                             'lr_cf':['beta',[1,1]],
#                             'lr_pred':['beta',[1,1]]}
						 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_justLR'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(feature_learner_2states_justLR)
# beta_mf':['gamma',[0.29,16.65]],
#                             'lr_neg':['beta',[0.18,0.75]],
#                             'safe_b':['gamma',[0.67,4.52]],
#                             'pred_b':['gamma',[0.36,10.71]],
#                             'lr_val':['beta',[0.25,0.62]],
#                             'pre_change':['norm',[-0.1,1]],
#                             're_change':['norm',[0,1,1]],
#                             'lr_cf':['beta',[0.21,1.11]],
#                             'beta_rfi':['gamma',[0.61,1.89]]
name_1='feature_learner_2states_rfiGamma_stickyNorm_sim'

# final priors from pilot data
# final param info: [['beta_mf', 'gamma', [0.8971618669572339, 1.2578395737197425]], 
# ['lr_neg', 'beta', [0.19732339190853893, 0.9930987340254532]], 
# ['safe_b', 'gamma', [0.7835403181803373, 4.692371637140057]], 
# ['pred_b', 'gamma', [0.27745505791599706, 11.669023947625934]], 
# ['lr_val', 'beta', [0.4638727122126635, 0.7711522625129456]], 
# ['pre_change', 'norm', [-0.20999312690831076, 1.2592965458863463]], 
# ['re_change', 'norm', [0.21174351758494878, 0.8294144353420907]], 
# ['lr_cf', 'beta', [0.18338783380358722, 0.9393123257617776]], 
# ['beta_rfi_rew', 'norm', [0.744260412514038, 1.6964288950349944]], 
# ['beta_rfi_pred', 'norm', [-1.4395915824743035, 2.004467549462906]]]
							#name of parameters and hyperpriors
group_parameters_info_mood={'beta_mf':['gamma',[1, 1]],
							'lr_neg':['beta',[1,1]],
							'safe_b':['gamma',[1, 1]],
							'pred_b':['gamma',[1, 1]],
							'lr_val':['beta',[1, 1]],
							'pre_change':['norm',[0, 1]], 
							're_change':['norm',[0, 1]],
							'lr_cf':['beta',[1,1]],
							'beta_rfi_rew':['gamma',[1,1]],
							'beta_rfi_pred':['gamma',[1,1]],
							# 'rpre_change':['norm',[0, 1]], 
							# 'rre_change':['norm',[0, 1]],
							'stick':['norm',[0,1]],
							# 'p_sens':['gamma',[1,1]]
							}
							# 'lr_cfac_p':['beta',[1,1]]} 
likelihood='feature_learner_lik_beta_counterfactual_two_change_2iR_sticky'
build_model(name_1,likelihood,group_parameters_info_mood,number_subjects,parameter_sample_size)
all_models.append(feature_learner_2states_rfiGamma_stickyNorm_sim)

# name_1='feature_learner_2states_rfiGamma_stickyNorm_sim_noGP'

# group_parameters_info_mood={'beta_mf':['gamma',[1, 1]],
# 							'lr_neg':['beta',[1,1]],
# 							'safe_b':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'lr_val':['beta',[1, 1]],
# 							'pre_change':['norm',[0, 1]], 
# 							're_change':['norm',[0, 1]],
# 							'lr_cf':['beta',[1,1]],
# 							# 'rpre_change':['norm',[0, 1]], 
# 							# 'rre_change':['norm',[0, 1]],
# 							'stick':['norm',[0,1]],
# 							# 'p_sens':['gamma',[1,1]]
# 							}
# 							# 'lr_cfac_p':['beta',[1,1]]} 
# likelihood='feature_learner_lik_beta_counterfactual_two_change_2iR_sticky_noGP'
# build_model(name_1,likelihood,group_parameters_info_mood,number_subjects,parameter_sample_size)
# all_models.append(feature_learner_2states_rfiGamma_stickyNorm_sim_noGP)

# # Standard model-free Q-learner


# name_1='feature_learner_2states_justLR'
# group_parameters_info_m={'be_mf':['gamma',[1,2]],
#                             'lr_safe':['beta',[2,2]],
#                             'betaf':['gamma',[1,4]],
#                             'lr_val':['beta',[2,2]],
#                             'lr_cf':['beta',[2,4]],
#                             'lr_safe':['beta',[2,2]]}
						 
# likelihood='feature_learner_lik_beta_counterfactual_two_empirical_justLR'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(feature_learner_2states)

# In[12]:




 # Fit models




#Partition data for parallelization
x=np.arange(number_subjects)
cores_subs=12
bins=np.arange(0,number_subjects-1,number_subjects/cores_subs)
subjects_partitioned=[]
for i in range(1,cores_subs+1):
	subjects_partitioned.append(x[np.digitize(x,bins)==i])
print('subjects partitioned : {}'.format(subjects_partitioned))
print(len(subjects_partitioned))

#divide up likelihood over all cores
x=np.arange(parameter_sample_size)
cores=4
bins=np.arange(0,parameter_sample_size-1,parameter_sample_size/cores)
samples_partitioned=[]
for i in range(1,cores+1):
	samples_partitioned.append(x[np.digitize(x,bins)==i])


	
import concurrent.futures
from multiprocessing.dummy import Pool
import multiprocessing
from itertools import repeat,starmap
import numpy as np
import time


#current_dataset=all_data_input #make sure dataset is correct before running model fitting!!
current_dataset=all_data3

#iterate over models
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
		
		#Compute iBIC
		Nparams = 2*len(parameter_names)
		Ndatapoints = float(number_subjects*num_trials) #total number of datapoints
		exec('total_evidence=sum([x[1][0] for x in all_results_{}])'.format(model.name))
		new_bic = -2.0*float(total_evidence) + Nparams*np.log(Ndatapoints) # Bayesian Information Criterion
		improvement = old_bic - new_bic # compute improvement of fit
		
		#only retain evidence and BIC if they improve
		if improvement > 0:
			model.model_evidence_group=total_evidence
			model.bic=new_bic
		
		#read out latest iteration
		print('{}- iBIC old:{}, new: {}\n'.format(model.name, old_bic, new_bic))

print('')
print('final param info:')
print(parameter_info)
print('')




# 'beta_mf':['gamma',[1, 1]],
# 							'lr_neg':['beta',[1,1]],
# 							'safe_b':['gamma',[1, 1]],
# 							'pred_b':['gamma',[1, 1]],
# 							'lr_val':['beta',[1, 1]],
# 							'pre_change':['norm',[0, 1]], 
# 							're_change':['norm',[0, 1]],
# 							'lr_cf':['beta',[1,1]],
# 							'beta_rfi':['norm',[0,1]],
# 							'beta_rfi_pred':['norm',[0,1]],
# 							# 'rpre_change':['norm',[0, 1]], 
# 							# 'rre_change':['norm',[0, 1]],
# 							'stick':['norm',[0,1]]
from scipy.stats import pearsonr as corr
import seaborn as sns
import matplotlib.pyplot as plt

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
lr_neg=[]
lr_pos=[]
beta_rfis_p=[]
beta_rfis_r=[]
beta_rfis_pc=[]
beta_rfis_rc=[]
beta_st=[]
lr_st=[]
psens_f=[]
for subject in range(number_subjects):
   zero.append(all_results[subject][0])
   one.append(all_results[subject][1])
   two.append(all_results[subject][2][1])
   lr_neg.append(all_results[subject][3][1])
   beta_s.append(all_results[subject][4][1])
   beta_p.append(all_results[subject][5][1])
   pre_change.append(all_results[subject][7][1])
   re_change.append(all_results[subject][8][1])
   lr_pos.append(all_results[subject][9][1])
   beta_rfis_r.append(all_results[subject][10][1])
   beta_rfis_p.append(all_results[subject][11][1])
   # beta_rfis_rc.append(all_results[subject][10][1])
   # beta_rfis_pc.append(all_results[subject][11][1])
   beta_st.append(all_results[subject][12][1])
   # psens_f.append(all_results[subject][13][1])


print('')
print(zero)
print('')

print(one)
print('')

print('beta mf betas: {}'.format(two))
print('')

print('lr features: mean{}, sd: {}'.format(np.mean(lr_neg),np.std(lr_neg)))
print(lr_neg)
print('')

print('lr cf: mean{}, sd: {}'.format(np.mean(lr_pos),np.std(lr_pos)))
print(lr_pos)
print('')

print('feature betas safe: mean{}, sd: {}'.format(np.mean(beta_s),np.std(beta_s)))
print(beta_s)
print('')

print('feature betas pred: mean{}, sd: {}'.format(np.mean(beta_p),np.std(beta_p)))
print(beta_p)
print('')

print('rew change: mean{}, sd: {}'.format(np.mean(re_change),np.std(re_change)))
print(re_change)
print('')

print('pre change: mean{}, sd: {}'.format(np.mean(pre_change),np.std(pre_change)))
print(pre_change)
print('')

print('feature betas RFI rew: mean{}, sd: {}'.format(np.mean(beta_rfis_r),np.std(beta_rfis_r)))
print(beta_rfis_r)
print('')

print('feature betas RFI pred: mean{}, sd: {}'.format(np.mean(beta_rfis_p),np.std(beta_rfis_p)))
print(beta_rfis_p)
print('')

# print('feature betas RFI reward CHANGE: mean{}, sd: {}'.format(np.mean(beta_rfis_rc),np.std(beta_rfis_rc)))
# print(beta_rfis_rc)
# print('')

# print('feature betas RFI pred CHANGE: mean{}, sd: {}'.format(np.mean(beta_rfis_pc),np.std(beta_rfis_pc)))
# print(beta_rfis_pc)
# print('')

print('feature betas sticky: mean{}, sd: {}'.format(np.mean(beta_st),np.std(beta_st)))
print(beta_st)
print('')


#,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,beta_irf)

# #FOR SIMULATED DATA ONLY -- PARAMETER RECOVERY ESTIMATES
print('')
r,p=corr(lr_neg,lrf_safe)
print("correlation of {} for LR features, p value {}\n".format(r,p))
print('')

r,p=corr(beta_s,beta_safe)
print("correlation of {} for safes, p value {}\n".format(r,p))
print('')
r,p=corr(beta_p,beta_pred)
print("correlation of {} for preds, p value {}".format(r,p))
print('')
r,p=corr(pre_change,pred_change)
print("correlation of {} for change in pred, p value {}\n".format(r,p))
print('')

r,p=corr(re_change,rew_change)
print("correlation of {} for change in rew, p value {}\n".format(r,p))
print('')
r,p=corr(lr_pos,lr_cfr)
print("correlation of {} for cf lr {}\n".format(r,p))
print('')

r,p=corr(rfi_rews,beta_rfis_r)
print("correlation of {} for RFI REW {}\n".format(r,p))
print('')

r,p=corr(rfi_preds,beta_rfis_p)
print("correlation of {} for RFI PRED {}\n".format(r,p))
print('')

r,p=corr(stick,beta_st)
print("correlation of {} for STICKINESS {}\n".format(r,p))
print('')


# r,p=corr(psens_f,psenss)
# print("correlation of {} for p sensitivity {}\n".format(r,p))
# print('')
# r,p=corr(lr_sticks,lr_st)
# print("correlation of {} for LR STICKINESS {}\n".format(r,p))
# print('')

# r,p=corr(psens_f,beta_p)
# print("correlation of {} psens and pred beta both fitted {}\n".format(r,p))
# print('')





# # # Generate a large random dataset
# d = pd.DataFrame()
# d['lr_feature_true']=lrf_safe
# d['beta_safe_true']=beta_safe
# d['beta_pred_true']=beta_pred
# d['rew_change_true']=rew_change
# d['pred_change_true']=pred_change
# d['cf_lr_true']=lr_cfr
# d['beta_irf_REW']=rfi_rews
# d['beta_irf_PRED']=rfi_preds
# d['beta_stick']=stick
# # d['psens']=psenss
# d['lr_feature_sim']=lr_neg
# d['beta_safe_sim']=beta_s
# d['beta_pred_sim']=beta_p
# d['rew_change_sim']=re_change
# d['pred_change_sim']=pre_change
# d['cf_lr_sim']=lr_pos
# d['beta_irf_REW_sim']=beta_rfis_r
# d['beta_irf_PRED_sim']=beta_rfis_p
# d['beta_stick_sim']=beta_st
# # d['psens_sim']=psens_f

# # Compute the correlation matrix
# corr = d.corr()

# corr.to_csv('correlations_2irf_ignoredRF.csv')

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, cmap=cmap, center=0,
#             square=True, linewidths=.5)

# plt.show()