from numpy.random import choice
from random import random
from random import shuffle
import scipy.special as sf
import numpy as np
import pandas as pd
import scipy as sp
import os

def simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(subject_dfs):
	from numpy.random import choice as choice_draw
	from random import random
	from random import shuffle
	from random import sample
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp	

	#define hyperparameters for each group distribution and sample from them
	all_lr_values=[]
	all_lr_features=[]
	all_datasets=[]
	all_outcomes=[]
	q_rew_all=[]
	num_subjects=len(subject_dfs)
	#retrieve samples of parameters from model
	for subject in range(num_subjects):
		data=subject_dfs[subject]
		for game_played in range(1):
			#initialize variables for each subjects' data
			q_values=np.zeros((2,)) #initialize values for each sample, assumes 2 choices
			last_choice=np.zeros((2,))
			feature_matrix=np.zeros((2,2))
			lrf0=0.12
			lrf1=0.12
			lr_features=np.array((lrf0,lrf1))
			lr_cf_r=0.12
			lr_cf_p=0.12
			last_beta=0
			lr_value=0
			
			#collect subject level beta weights
			beta_safe=100
			beta_pred=100
			pred_changes=0
			rew_changes=0
			# betas_rew=np.array((beta_rew[subject],beta_rew[subject]))
			# betas_punish=np.array((beta_punish[subject],beta_punish[subject]))
			beta_weights_SR=np.array((beta_safe,beta_pred))
			beta_mf=0
			rfi_rew=0
			rfi_pred=0
			switching=[]
			choices=[]
			other_choices=[]
			outcomes=[]
			accurates=[]
			q_rews=[]
			safes=[]
			preds=[]

			#account for data missing
	
			errors=data[data['chosen_state']=='SLOW']
			errors=errors.reset_index()
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
				reward_function_timeseries=[[data.f_1_reward[i],data.f_2_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
				outcome=data.reward_received
				safes=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
				predators=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

			elif reverse=='yes':
				reward_function_timeseries=[[data.f_2_reward[i],data.f_1_reward[i]] for i in range(len(data.s_2_f_1_outcome))]
				outcome=data.reward_received
				safes=[[data.s_1_f_2_outcome[i],data.s_2_f_2_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]
				predators=[[data.s_1_f_1_outcome[i],data.s_2_f_1_outcome[i]] for i in range(len(data.s_2_f_1_outcome))]

			for trial in range(len(choice)):
	
				# participant is told which predator they're facing
				current_reward_function=reward_function_timeseries[trial]

				#weighting is a combo of beta weights and reward function
				full_weights=beta_weights_SR*current_reward_function
				#derive SR q-values by multiplying feature vectors by reward function
				q_sr=np.matmul(feature_matrix,full_weights)

				q_irf_weights=np.array((rfi_rew,rfi_pred))

				q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)
				#integrated q_values from MF and SR systems
				q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF +((last_beta)*(last_choice))

				#determine choice via softmax
				action_values = np.exp((q_integrated)-sf.logsumexp(q_integrated));
				values=action_values.flatten()
				draw = choice_draw(values,1,p=values)[0]
				indices = [i for i, x in enumerate(values) if x == draw]
				current_choice=sample(indices,1)[0]
				subject_choice=int(choice[trial])-1
				if current_choice==subject_choice:
					accurates.append(1)
				else:
					accurates.append(0)


				
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
			 
				#get feature outcomes for current trial
				feature_outcomes=np.array(([safes[trial][0],predators[trial][0]],[safes[trial][1],predators[trial][1]]))
				safes.append(list(feature_outcomes[:,0]))                    
				preds.append(list(feature_outcomes[:,1])) 
				
				#concatenate all feature outcomes into single array
				current_features=feature_outcomes[current_choice]
				
				#determine current outcome by feature vector * reward function
				current_outcome=sum(current_features*current_reward_function)

				#save how distant estimated EV is from actual EV 
				# q_rews.append(np.abs(current_bandit-(q_sr[current_choice])))
				outcomes.append(current_outcome)
				
				# Prediction error over features
				pe_features = feature_outcomes-feature_matrix # Feature prediction errors
				#Feature updating
				current_decision=current_choice
				last_choice=np.zeros((2,))
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
			all_outcomes.append(np.sum(accurates)/len(choice))

			
		
	return np.mean(all_outcomes),np.std(all_outcomes), all_outcomes


# #### collect subject data

subject_data=[pd.read_csv(sub) for sub in os.listdir(os.curdir) if sub.startswith('sub_3')]

accurates,sd_s,outs=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(subject_data)

print('mean accuracy: {}, SD accuracy: {}\n all outcomes: {}'.format(accurates,sd_s,outs))

