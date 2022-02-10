def simulate_one_step_featurelearner_beta_parallel(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_mild=params[5]
	beta_pred=params[6]
	mf_betas=params[7]
	bandits=params[8]
	reward_function_time_series=params[9]
	games_played=params[10]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf,lrf))
	lr_value=lrv
	beta_weights_SR=np.array((beta_safe,beta_mild,beta_pred))
	beta_mf=mf_betas
	all_outcomes=[]
	

	for i in range(games_played):
		q_values=np.zeros((3,))
		feature_vector=np.zeros((3,3))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		milds=[]
		predators=[]
		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		for trial in range(num_trials):

			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial] 
			
			#weighting is a combo of beta weights and reward function
			full_weights=beta_weights_SR*current_reward_function

			#derive SR q-values by multiplying feature vectors by reward function
			q_sr=np.matmul(feature_vector,full_weights) 
			
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
			choices.append(current_choice)
		 
			#get current feature latent probabilities
			cb1=bandits[int(current_choice)][0][trial]
			cb2=bandits[int(current_choice)][1][trial]
			cb3=bandits[int(current_choice)][2][trial]
			current_bandit=np.array((cb1,cb2,cb3))
			#print(np.random.binomial(1,[current_bandit][0]))
			#get feature outcomes for current trial
			feature_outcome1=np.random.binomial(1,[current_bandit][0])[0]
			safes.append(feature_outcome1)
			feature_outcome2=np.random.binomial(1,[current_bandit][0])[1]
			milds.append(feature_outcome2)
			feature_outcome3=np.random.binomial(1,[current_bandit][0])[2]
			predators.append(feature_outcome3)
			
			#concatenate all feature outcomes into single array
			current_features=np.array((feature_outcome1,feature_outcome2,feature_outcome3))
			
			#determine current outcome by feature vector * reward function
			current_outcome=sum(current_features*current_reward_function)

			#save reward
			outcomes.append(current_outcome)
			
			# Prediction error over features
			pe_features = current_features-feature_vector[current_choice] # Feature prediction errors
			
			#Feature updating
			feature_vector[current_choice] = feature_vector[current_choice]+ (lr_features*pe_features)
			
			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))
		
		all_outcomes.append(np.sum(outcomes))

	return [np.mean(all_outcomes),np.std(all_outcomes)]



def simulate_one_step_featurelearner_counterfactual_parallel(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_mild=params[5]
	beta_pred=params[6]
	mf_betas=params[7]
	bandits=params[8]
	reward_function_time_series=params[9]
	games_played=params[10]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf,lrf))
	lr_value=lrv
	beta_weights_SR=np.array((beta_safe,beta_mild,beta_pred))
	beta_mf=mf_betas
	all_outcomes=[]
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((3,))
		feature_matrix=np.zeros((3,3))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		milds=[]
		predators=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		for trial in range(num_trials):
					
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
			
			
			#save choice   
			choices.append(current_choice)
		 
			#get current feature latent probabilities
			#get current feature latent probabilities

			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
			cb3=[x[trial] for x in bandits[2]]

			current_bandit=np.array((cb1,cb2,cb3))

#                     print(np.random.binomial(1,[current_bandit][0]))
			#get feature outcomes for current trial
	   
			feature_outcomes=np.random.binomial(1,current_bandit)
			safes.append(list(feature_outcomes[:,0]))                    
			milds.append(list(feature_outcomes[:,1])) 
			predators.append(list(feature_outcomes[:,2])) 
			
			#concatenate all feature outcomes into single array
			current_features=feature_outcomes[current_choice]
			
			#determine current outcome by feature vector * reward function
			current_outcome=sum(current_features*current_reward_function)
			#save how distant estimated EV is from actual EV 
			outcomes.append(current_outcome)
			
			# Prediction error over features
			pe_features = feature_outcomes-feature_matrix # Feature prediction errors
			#Feature updating
			feature_matrix = feature_matrix+ (lr_features*pe_features)
			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)]


def simulate_one_step_featurelearner_counterfactual_parallel_2states(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	mf_betas=params[6]
	bandits=params[7]
	reward_function_time_series=params[8]
	games_played=params[9]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	lr_value=lrv
	beta_weights_SR=np.array((beta_safe,beta_pred))
	beta_mf=mf_betas
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		predators=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		for trial in range(num_trials):
					
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
			
			
			#save choice   
			choices.append(current_choice)
		 
			#get current feature latent probabilities
			#get current feature latent probabilities

			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]

			current_bandit=np.array((cb1,cb2))

#                     print(np.random.binomial(1,[current_bandit][0]))
			#get feature outcomes for current trial
	   
			feature_outcomes=np.random.binomial(1,current_bandit)
			safes.append(list(feature_outcomes[:,safe_index]))                    
			predators.append(list(feature_outcomes[:,predator_index])) 
			
			#concatenate all feature outcomes into single array
			current_features=feature_outcomes[current_choice]
			
			#determine current outcome by feature vector * reward function
			current_outcome=sum(current_features*current_reward_function)

			#save how distant estimated EV is from actual EV 
			outcomes.append(current_outcome)
			
			# Prediction error over features
			pe_features = feature_outcomes-feature_matrix # Feature prediction errors
			#Feature updating
			feature_matrix = feature_matrix+ (lr_features*pe_features)
			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lr_value*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)]


def simulate_2iRF_sticky(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	beta_mf=params[6]
	lr_cf=lrf
	last_beta=params[8]
	pred_changes=params[9]
	rew_changes=params[10]
	rfi_rew=params[11]
	rfi_pred=params[12]
	bandits=params[13]
	reward_function_time_series=params[14]
	games_played=params[15]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	beta_weights_SR=np.array((beta_safe,beta_pred))
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
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
			  
					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]
			# current_reward_function=np.flip(current_reward_function)

			#weighting is a combo of beta weights and reward function
			full_weights=beta_weights_SR*current_reward_function
			#derive SR q-values by multiplying feature vectors by reward function
			q_sr=np.matmul(feature_matrix,full_weights)

			q_irf_weights=np.array((rfi_rew,-1.0*rfi_pred))
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
			
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
	
			current_bandit=np.array((cb1,cb2))
 
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
			#safe feature
			feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lrf*pe_features[current_decision,0])
			feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
			#predator feature updating
			feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lrf*pe_features[current_decision,1])
			feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lrv*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)]

def simulate_2iRF_sticky_flipped(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	beta_mf=params[6]
	lr_cf=params[7]
	last_beta=params[8]
	pred_changes=params[9]
	rew_changes=params[10]
	rfi_rew=params[11]
	rfi_pred=params[12]
	bandits=params[13]
	reward_function_time_series=params[14]
	games_played=params[15]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	beta_weights_SR=np.array((beta_safe,beta_pred))
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
					
		for trial in range(num_trials):
			if trial>=80:
				beta_weights_SR=np.array((beta_pred-pred_changes,beta_safe-rew_changes))
				
			else:
				beta_weights_SR=np.array((beta_pred+pred_changes,beta_safe+rew_changes))
			  
					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]
			# current_reward_function=np.flip(current_reward_function)

			#weighting is a combo of beta weights and reward function
			full_weights=beta_weights_SR*current_reward_function
			#derive SR q-values by multiplying feature vectors by reward function
			q_sr=np.matmul(feature_matrix,full_weights)

			q_irf_weights=np.array((-1.0*rfi_pred,rfi_rew))
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
			
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
	
			current_bandit=np.array((cb1,cb2))
 
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
			#safe feature
			feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lrf*pe_features[current_decision,0])
			feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
			#predator feature updating
			feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lrf*pe_features[current_decision,1])
			feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lrv*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)]

def simulate_2iRF_sticky_pflipped(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	beta_mf=params[6]
	lr_cf=params[7]
	last_beta=params[8]
	pred_changes=params[9]
	rew_changes=params[10]
	rfi_rew=params[11]
	rfi_pred=params[12]
	bandits=params[13]
	reward_function_time_series=params[14]
	games_played=params[15]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	beta_weights_SR=np.array((beta_safe,beta_pred))
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
					
		for trial in range(num_trials):
			if trial>=80:
				beta_weights_SR=np.array((beta_pred-pred_changes,beta_safe-rew_changes))
				
			else:
				beta_weights_SR=np.array((beta_pred+pred_changes,beta_safe+rew_changes))
			  
					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]
			# current_reward_function=np.flip(current_reward_function)

			#weighting is a combo of beta weights and reward function
			full_weights=beta_weights_SR*current_reward_function
			#derive SR q-values by multiplying feature vectors by reward function
			q_sr=np.matmul(feature_matrix,full_weights)

			q_irf_weights=np.array((-1.0*rfi_pred,rfi_rew))
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
			
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
	
			current_bandit=np.array((cb1,cb2))
 
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
			if current_outcome==-1.0:
				outcomes.append(current_outcome)
			
			# Prediction error over features
			pe_features = feature_outcomes-feature_matrix # Feature prediction errors
			#Feature updating
			current_decision=current_choice
			last_choice=np.zeros((num_bandits,))
			last_choice[current_decision]=1.0

			other_decision=abs(current_choice-1)
			#safe feature
			feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lrf*pe_features[current_decision,0])
			feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
			#predator feature updating
			feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lrf*pe_features[current_decision,1])
			feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lrv*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)]

def simulate_2iRF_sticky_rflipped(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	beta_mf=params[6]
	lr_cf=params[7]
	last_beta=params[8]
	pred_changes=params[9]
	rew_changes=params[10]
	rfi_rew=params[11]
	rfi_pred=params[12]
	bandits=params[13]
	reward_function_time_series=params[14]
	games_played=params[15]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	beta_weights_SR=np.array((beta_safe,beta_pred))
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
					
		for trial in range(num_trials):
			if trial>=80:
				beta_weights_SR=np.array((beta_pred-pred_changes,beta_safe-rew_changes))
				
			else:
				beta_weights_SR=np.array((beta_pred+pred_changes,beta_safe+rew_changes))
			  
					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]
			# current_reward_function=np.flip(current_reward_function)

			#weighting is a combo of beta weights and reward function
			full_weights=beta_weights_SR*current_reward_function
			#derive SR q-values by multiplying feature vectors by reward function
			q_sr=np.matmul(feature_matrix,full_weights)

			q_irf_weights=np.array((-1.0*rfi_pred,rfi_rew))
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
			
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
	
			current_bandit=np.array((cb1,cb2))
 
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
			if current_outcome==1.0:
				outcomes.append(current_outcome)
			
			# Prediction error over features
			pe_features = feature_outcomes-feature_matrix # Feature prediction errors
			#Feature updating
			current_decision=current_choice
			last_choice=np.zeros((num_bandits,))
			last_choice[current_decision]=1.0

			other_decision=abs(current_choice-1)
			#safe feature
			feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lrf*pe_features[current_decision,0])
			feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
			#predator feature updating
			feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lrf*pe_features[current_decision,1])
			feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lrv*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)]

def simulate_2iRF_sticky_punishments(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	beta_mf=params[6]
	lr_cf=params[7]
	last_beta=params[8]
	pred_changes=params[9]
	rew_changes=params[10]
	rfi_rew=params[11]
	rfi_pred=params[12]
	bandits=params[13]
	reward_function_time_series=params[14]
	games_played=params[15]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	beta_weights_SR=np.array((beta_safe,beta_pred))
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
		# num_puns=0			
		for trial in range(num_trials):
			# if trial>=80:
			# 	beta_weights_SR=np.array((beta_safe,beta_pred))
				
			# else:
			# 	beta_weights_SR=np.array((beta_safe,beta_pred))
			  
			beta_weights_SR=np.array((beta_safe,beta_pred))
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]
			# current_reward_function=np.flip(current_reward_function)

			# feature_matrix[1,0]=1
			# feature_matrix[0,0]=0
			#predator feature updating
			# feature_matrix[1,1]=1
			# feature_matrix[0,1]=0
			#weighting is a combo of beta weights and reward function
			full_weights=beta_weights_SR*current_reward_function

			#derive SR q-values by multiplying feature vectors by reward function
			q_sr=np.matmul(feature_matrix,full_weights)
			
			q_irf_weights=np.array((rfi_rew,-1*rfi_pred))
			q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)
			
			#integrated q_values from MF and SR systems
			q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF +((last_beta)*(last_choice))
			
			# if -1 in current_reward_function:
			# 	num_puns+=1
			# 	print(q_sr)
			# 	print(q_insensitive_RF)
			# 	print('q int {}'.format(q_integrated))

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
			
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
	
			current_bandit=np.array((cb1,cb2))
 
			#get feature outcomes for current trial
			feature_outcomes=np.random.binomial(1,current_bandit)
			safes.append(list(feature_outcomes[:,0]))
			preds.append(list(feature_outcomes[:,1])) 
   
			#concatenate all feature outcomes into single array
			current_features=feature_outcomes[current_choice]
			# print('current Rf {}'.format(current_reward_function))
			#determine current outcome by feature vector * reward function
			# print('current features {}'.format(current_features))
			current_outcome=sum(current_features*current_reward_function)
			# print('current outcome {}'.format(current_outcome))
			#save how distant estimated EV is from actual EV 
			q_rews.append(np.abs(current_bandit-(q_sr[current_choice])))
			if current_outcome==-1.0:
				outcomes.append(current_outcome)

	
			# Prediction error over features
			pe_features = feature_outcomes-feature_matrix # Feature prediction errors
			#Feature updating
			current_decision=current_choice
			last_choice=np.zeros((num_bandits,))
			last_choice[current_decision]=1.0

			other_decision=abs(current_choice-1)
			#safe feature
			feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lrf*pe_features[current_decision,0])
			feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
			#predator feature updating
			feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lrf*pe_features[current_decision,1])
			feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lrv*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))

		# print('num puns {}'.format(num_puns))
	return [np.mean(all_outcomes),np.std(all_outcomes)]


def simulate_2iRF_sticky_rewards(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp

	num_subjects=params[0]
	num_trials=params[1]
	lrf=params[2]
	lrv=params[3]
	beta_safe=params[4]
	beta_pred=params[5]
	beta_mf=params[6]
	lr_cf=params[7]
	last_beta=params[8]
	pred_changes=params[9]
	rew_changes=params[10]
	rfi_rew=params[11]
	rfi_pred=params[12]
	bandits=params[13]
	reward_function_time_series=params[14]
	games_played=params[15]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	lr_features=np.array((lrf,lrf))
	beta_weights_SR=np.array((beta_safe,beta_pred))
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]


		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
					
		for trial in range(num_trials):
			if trial<80:
				beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
				
			else:
				beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))
			  
					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]
			# current_reward_function=np.flip(current_reward_function)

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
			
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
	
			current_bandit=np.array((cb1,cb2))
 
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
			if current_outcome==1:
				outcomes.append(current_outcome)
			
			# Prediction error over features
			pe_features = feature_outcomes-feature_matrix # Feature prediction errors
			#Feature updating
			current_decision=current_choice
			last_choice=np.zeros((num_bandits,))
			last_choice[current_decision]=1.0

			other_decision=abs(current_choice-1)
			#safe feature
			feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lrf*pe_features[current_decision,0])
			feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
			#predator feature updating
			feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lrf*pe_features[current_decision,1])
			feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

			#Model-Free action value updating
			q_values[current_choice] = q_values[current_choice]+ (lrv*(current_outcome-q_values[current_choice]))

		all_outcomes.append(np.sum(outcomes))
  
	return [np.mean(all_outcomes),np.std(all_outcomes)/2000]



import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#binary VKF filter designed for volatile environments -- Piray & Daw 2020

def simulate_VKF_binary(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp
	from scipy.stats import bernoulli
	num_subjects=params[0]
	num_trials=params[1]
	omega1=params[2]
	# beta_weight=100
	omega2=omega1	
	omega3=omega1
	omega4=omega1
	bandits=params[3]
	volatility=0.2
	volatility2=0.05
	reward_function_time_series=params[4]
	games_played=params[5]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]



		#prior on means of feature probabilities
		mean_a1r=0
		mean_a2r=0
		mean_a1p=0
		mean_a2p=0



		#prior on mean volatilities
		variance_a1r=10
		variance_a2r=10
		variance_a1p=10
		variance_a2p=10



		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
					
		for trial in range(num_trials):



			#compute kalman gain on variances
			k1=(variance_a1r+volatility)/(variance_a1r+volatility+omega1)
			k2=(variance_a2r+volatility)/(variance_a2r+volatility+omega1)
			k3=(variance_a1p+volatility2)/(variance_a1p+volatility2+omega1)
			k4=(variance_a2p+volatility2)/(variance_a2p+volatility2+omega1)

			#compute learning rates
			alpha1=np.sqrt(variance_a1r+volatility)
			alpha2=np.sqrt(variance_a2r+volatility)
			alpha3=np.sqrt(variance_a1p+volatility2)
			alpha4=np.sqrt(variance_a2p+volatility2)

			#transformed predictions
			prediction_mean_a1r=sigmoid(mean_a1r)
			prediction_mean_a2r=sigmoid(mean_a2r)
			prediction_mean_a1p=sigmoid(mean_a1p)
			prediction_mean_a2p=sigmoid(mean_a2p)


			
			prediction_array=np.array(([prediction_mean_a1r,prediction_mean_a2r],[prediction_mean_a1p,prediction_mean_a2p]))

					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]

			

	
			# current_choice=np.argmax(current_action_values)

			
		 
			#get current feature latent probabilities
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
			current_bandit=np.array((cb1,cb2))
			#derive SR q-values by multiplying feature vectors by reward function
			
			# #compute action values -- oracle model
			# current_action_values=np.matmul(current_bandit,current_reward_function)

			# #compute action values -- ideal observer (VKF) model
			current_action_values=np.matmul(prediction_array,current_reward_function)

			#softmax decision rule
			# action_values = np.exp((beta_weight*current_action_values)-sf.logsumexp(beta_weight*current_action_values));
			# values=action_values.flatten()
			# draw = choice(values,1,p=values)[0]
			# indices = [i for i, x in enumerate(values) if x == draw]
			# current_choice=sample(indices,1)[0]

			#argmax decision rule
			current_choice=np.argmax(current_action_values)
	
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
			feature_outcomes=np.random.binomial(1,current_bandit)
			safes.append(list(feature_outcomes[:,0]))
			preds.append(list(feature_outcomes[:,1])) 
			outcome_a1r=feature_outcomes[0,0]
			outcome_a2r=feature_outcomes[1,0]
			outcome_a1p=feature_outcomes[0,1]
			outcome_a2p=feature_outcomes[1,1]


   
			#concatenate all feature outcomes into single array
			current_features=feature_outcomes[current_choice]
			
			#determine current outcome by feature vector * reward function
			current_outcome=sum(current_features*current_reward_function)
			outcomes.append(current_outcome)
			
			#Prediction Errors
			pe_a1r=outcome_a1r-prediction_mean_a1r
			pe_a2r=outcome_a2r-prediction_mean_a2r
			pe_a1p=outcome_a1p-prediction_mean_a1p
			pe_a2p=outcome_a2p-prediction_mean_a2p

			#update means of feature probabilities
			mean_a1r=mean_a1r+(alpha1*pe_a1r)
			mean_a2r=mean_a2r+(alpha2*pe_a2r)
			mean_a1p=mean_a1p+(alpha3*pe_a1p)
			mean_a2p=mean_a2p+(alpha4*pe_a2p)

			#update variances
			variance_a1r=(1-k1)*(variance_a1r+volatility)
			variance_a2r=(1-k2)*(variance_a2r+volatility)
			variance_a1p=(1-k3)*(variance_a1p+volatility2)
			variance_a2p=(1-k4)*(variance_a2p+volatility2)
			

			


		all_outcomes.append(np.sum(outcomes))
	

	return [np.mean(all_outcomes),np.std(all_outcomes)]

def simulate_VKF_binary(params):
	from random import sample
	from numpy.random import choice
	from random import random
	from random import shuffle
	import scipy.special as sf
	import numpy as np
	import pandas as pd
	import scipy as sp
	from scipy.stats import bernoulli
	num_subjects=params[0]
	num_trials=params[1]
	omega1=params[2]
	# beta_weight=100
	omega2=omega1	
	omega3=omega1
	omega4=omega1
	bandits=params[3]
	volatility=0.2
	volatility2=0.05
	reward_function_time_series=params[4]
	games_played=params[5]
	num_bandits=len(bandits)     
   

	#initialize variables for each subjects' data
	all_outcomes=[]
	predator_index=1
	safe_index=0
	
	#retrieve samples of parameters from model
	for i in range(games_played):
		
		q_values=np.zeros((2,))
		feature_matrix=np.zeros((2,2))
		last_choice=np.zeros((2,))
		#outcomes to save
		switching=[]
		choices=[]
		outcomes=[]
		q_rews=[]
		safes=[]
		preds=[]



		#prior on means of feature probabilities
		mean_a1r=0
		mean_a2r=0
		mean_a1p=0
		mean_a2p=0



		#prior on mean volatilities
		variance_a1r=10
		variance_a2r=10
		variance_a1p=10
		variance_a2p=10



		#simulate decisions and outcomes based on feature learner that combines MF and SR Q-values
		
					
		for trial in range(num_trials):



			#compute kalman gain on variances
			k1=(variance_a1r+volatility)/(variance_a1r+volatility+omega1)
			k2=(variance_a2r+volatility)/(variance_a2r+volatility+omega1)
			k3=(variance_a1p+volatility2)/(variance_a1p+volatility2+omega1)
			k4=(variance_a2p+volatility2)/(variance_a2p+volatility2+omega1)

			#compute learning rates
			alpha1=np.sqrt(variance_a1r+volatility)
			alpha2=np.sqrt(variance_a2r+volatility)
			alpha3=np.sqrt(variance_a1p+volatility2)
			alpha4=np.sqrt(variance_a2p+volatility2)

			#transformed predictions
			prediction_mean_a1r=sigmoid(mean_a1r)
			prediction_mean_a2r=sigmoid(mean_a2r)
			prediction_mean_a1p=sigmoid(mean_a1p)
			prediction_mean_a2p=sigmoid(mean_a2p)


			
			prediction_array=np.array(([prediction_mean_a1r,prediction_mean_a2r],[prediction_mean_a1p,prediction_mean_a2p]))

					
			# participant is told which predator they're facing
			current_reward_function=reward_function_time_series[trial]

			

	
			# current_choice=np.argmax(current_action_values)

			
		 
			#get current feature latent probabilities
			cb1=[x[trial] for x in bandits[0]]
			cb2=[x[trial] for x in bandits[1]]
			current_bandit=np.array((cb1,cb2))
			#derive SR q-values by multiplying feature vectors by reward function
			
			# #compute action values -- oracle model
			# current_action_values=np.matmul(current_bandit,current_reward_function)

			# #compute action values -- ideal observer (VKF) model
			current_action_values=np.matmul(prediction_array,current_reward_function)

			#softmax decision rule
			# action_values = np.exp((beta_weight*current_action_values)-sf.logsumexp(beta_weight*current_action_values));
			# values=action_values.flatten()
			# draw = choice(values,1,p=values)[0]
			# indices = [i for i, x in enumerate(values) if x == draw]
			# current_choice=sample(indices,1)[0]

			#argmax decision rule
			current_choice=np.argmax(current_action_values)
	
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
			feature_outcomes=np.random.binomial(1,current_bandit)
			safes.append(list(feature_outcomes[:,0]))
			preds.append(list(feature_outcomes[:,1])) 
			outcome_a1r=feature_outcomes[0,0]
			outcome_a2r=feature_outcomes[1,0]
			outcome_a1p=feature_outcomes[0,1]
			outcome_a2p=feature_outcomes[1,1]


   
			#concatenate all feature outcomes into single array
			current_features=feature_outcomes[current_choice]
			
			#determine current outcome by feature vector * reward function
			current_outcome=sum(current_features*current_reward_function)
			outcomes.append(current_outcome)
			
			#Prediction Errors
			pe_a1r=outcome_a1r-prediction_mean_a1r
			pe_a2r=outcome_a2r-prediction_mean_a2r
			pe_a1p=outcome_a1p-prediction_mean_a1p
			pe_a2p=outcome_a2p-prediction_mean_a2p

			#update means of feature probabilities
			mean_a1r=mean_a1r+(alpha1*pe_a1r)
			mean_a2r=mean_a2r+(alpha2*pe_a2r)
			mean_a1p=mean_a1p+(alpha3*pe_a1p)
			mean_a2p=mean_a2p+(alpha4*pe_a2p)

			#update variances
			variance_a1r=(1-k1)*(variance_a1r+volatility)
			variance_a2r=(1-k2)*(variance_a2r+volatility)
			variance_a1p=(1-k3)*(variance_a1p+volatility2)
			variance_a2p=(1-k4)*(variance_a2p+volatility2)
			

			


		all_outcomes.append(np.sum(outcomes))
	

	return [np.mean(all_outcomes),np.std(all_outcomes)]
