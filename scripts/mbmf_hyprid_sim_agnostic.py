
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
            other_choices=[]
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
                other_choices.append(other_decision+1)
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
            exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["reward_received"]=outcomes'.format(subject,game_played))
            exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

            
        
    return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,predators
def simulate_one_step_featurelearner_counterfactual_2states(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds):
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
        rf1=np.load('rf_safe_80trials.npy')
        rf1=rf1/3.0
        rf2=np.load('rf_pred_80trials.npy')
        # rf1=np.flip(rf1,1)

        rf2=rf2/3.0


        reward_function_timeseries=np.concatenate((rf1,rf2),axis=0)
 
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
            rfi_rew=rfi_rews[subject]
            rfi_pred=rfi_preds[subject]
            # betas_rew=np.array((beta_rew[subject],beta_rew[subject]))
            # betas_punish=np.array((beta_punish[subject],beta_punish[subject]))
            beta_weights_SR=np.array((beta_safe,beta_pred))
            beta_mf=mf_betas[subject]
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
                # else:
                #     if trial<80:
                #         beta_weights_SR=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
                        
                #     else:
                #         beta_weights_SR=np.array((beta_safe+rew_changes,beta_pred+pred_changes))

                  


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
            exec('dataset_sub_{}_game_{}["preds"]=preds'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["safes"]=safes'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["reward_received"]=outcomes'.format(subject,game_played))
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
            exec('dataset_sub_{}_game_{}["preds"]=preds'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["safes"]=safes'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
            exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

            
        
    return all_datasets,np.mean(all_outcomes),np.mean(q_rew_all),choices,safes,preds

def simulate_one_step_featurelearner_MFRF(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfr,lr_cfp,rfi_rews,rfi_preds,st):
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
            q_values_rew=np.zeros((num_bandits,)) #initialize values for each sample, assumes 2 choices
            q_values_pred=np.zeros((num_bandits,))
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

                q_irf_weights=np.array((rfi_rew,-1*rfi_pred))
                q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)
                if 1 in current_reward_function:
                #integrated q_values from MF and SR systems
                    q_integrated=q_sr+((beta_mf)*(q_values_rew)) + q_insensitive_RF +((last_beta)*(last_choice))
                else:
                    q_integrated=q_sr+((beta_mf)*(q_values_pred)) + q_insensitive_RF +((last_beta)*(last_choice))
          

                        
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
                if 1 in current_reward_function:
                    q_values_rew[current_choice] = q_values_rew[current_choice]+ (lr_value*(current_outcome-q_values_rew[current_choice]))
                else:
                    q_values_pred[current_choice] = q_values_pred[current_choice]+ (lr_value*(current_outcome-q_values_pred[current_choice]))


            #save all output
            q_rew_all.append(np.sum(q_rews))
            all_outcomes.append(np.sum(outcomes))


            exec('dataset_sub_{}_game_{}["rf"]=list(reward_function_time_series)'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["preds"]=preds'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["safes"]=safes'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["choices"]=choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["outcomes"]=outcomes'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["other_choices"]=other_choices'.format(subject,game_played))
            exec('dataset_sub_{}_game_{}["switching"]=switching'.format(subject,game_played))
            exec('all_datasets.append(dataset_sub_{}_game_{})'.format(subject,game_played))

import numpy as np
from numpy.random import beta as betarnd
from numpy.random import gamma as gammarnd
from numpy.random import normal as normalrnd
from random import sample
from random import shuffle
import pandas as pd

import matplotlib


# df=pd.DataFrame()
# num_subjects=200
# num_trials=160
# lrv=list(np.zeros((num_subjects))+0.15)
# lrf=np.zeros((2,num_subjects))

# lrf_safe=list(np.zeros((num_subjects))+0.15)

# pred_change=list(np.zeros((num_subjects)))
# rew_change=list(np.zeros((num_subjects)))

# # lrf_mild=list(betarnd(2,5,num_subjects))
# lrf_each=list(betarnd(3,5,num_subjects))

# lr_cfr=list(np.zeros((num_subjects))+0.15)
# lr_cfp=list(np.zeros((num_subjects))+0.15)


# betas=np.zeros((2,num_subjects))

# beta_safe=list(np.zeros((num_subjects))+3)
# beta_pred=list(np.zeros((num_subjects))+3)
# betas[0]=beta_safe
# betas[1]=beta_pred
# #np.zeros((num_subjects))+3.0 # learn about predator feature

# #for beta model, there is only a single learning rate over features
# lrf2=np.zeros((2,num_subjects))
# lrf2[0]=lrf_safe # learnabout safe feature
# lrf2[1]=lrf_safe # learnabout safe feature
# # lrf2[2]=lrf_safe # learnabout safe feature



# mf_betas=list(np.zeros((num_subjects))+0)
# # beta_irf=list(gammarnd(0.29,3.92,num_subjects))
# rfi_rews=list(np.zeros((num_subjects))+0)
# rfi_preds=list(np.zeros((num_subjects))+6)
# stick=list(np.zeros((num_subjects))+0)


# bandits=np.load('safe_firsta.npy')
# # bandits2=np.load('safe_firsta.npy')
# # bandits=np.concatenate((bandits,bandits2),axis=2)
# # print(bandits.shape)

# rf2a=np.load('rf_safe_80trials.npy')
# rf2=np.flip(rf2a,1)*-1

# rf2a=rf2a/3.0
# rf2=rf2/3.0
# rf1a=np.load('rf_pred_80trials.npy')
# rf1=np.flip(rf1a,1)*-1
# rf1a=rf1a/3.0
# rf1=rf1/3.0

# reward_function_timeseries=np.concatenate((rf1,rf2),axis=0)
# reward_function_timeseries2=np.concatenate((rf1a,rf2a),axis=0)
# reward_function_timeseries3=np.concatenate((rf2a,rf1a),axis=0)
# reward_function_timeseries4=np.concatenate((rf2,rf1),axis=0)
# #CREATE SYNTHETIC DATA

# all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
# all_data4,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries2,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
# all_data5,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries3,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
# all_data6,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries4,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)

# all_data3=all_data3+all_data4+all_data5+all_data6
# #run logistic regression models on pilot data
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from scipy.stats import ttest_rel as tt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# import statsmodels.api as sm
# from math import isnan


# c_sub=0


# ppsP=[]
# p_nrfsP=[]
# np_nrfsP=[]
# np_psP=[]
# pps=[]
# p_nrfs=[]
# np_nrfs=[]
# np_ps=[]


# BppsP=[]
# Bp_nrfsP=[]
# Bnp_nrfsP=[]
# Bnp_psP=[]
# Bpps=[]
# Bp_nrfs=[]
# Bnp_nrfs=[]
# Bnp_ps=[]

# PPS_NON=[]
# PPS_CON=[]

# sub_num=1
# subject_names=[]
# corrs_pp_nrnrf=[]
# corrs_npp_nrnrf=[]

# current_dataset=all_data3
# for data_mf in current_dataset:

#     length_run=80 #just one environment
#     data_mf=data_mf.reset_index(drop=True)

#     order_var=1
#     first_var=1
#     second_var=-1
#     start=0
#     stop=160

    


 
#     p_values_safe=[]
#     p_values_pred=[]
#     coef_safe=[]
#     coef_pred=[]
#     failed=0
#     X_vars=pd.DataFrame()

    
#     data_mf=data_mf.iloc[start:stop]
#     start=0

#     data_mf=data_mf.reset_index(drop=True)
#     #design matrix multinomial logistic regression

#     choices=data_mf['choices']
#     # null_indices=np.where(pd.isnull(choices))[0]
#     data_mf_red=data_mf

#     pred_f=2
#     rew_f=1

#     X1=pd.DataFrame()
#     last_o_pred=[]
#     pred_rf=[]
#     env=[]
#     intercept=[]
#     order=[]
#     skipped=0
#     last_o_pred_other=[]
#     for trial in range(start,len(data_mf)):
#         order.append(order_var)
#         intercept.append(1)
#         if trial<80:
#             env.append(first_var)
#         else:
#             env.append(second_var)
        
#         chosen_state=data_mf_red['choices'][trial]
#         other_state=data_mf_red['other_choices'][trial]

#         last_o_pred_other.append(data_mf_red['preds'][trial][other_state-1])
#         last_o_pred.append(data_mf_red['preds'][trial][chosen_state-1])
#         pred_rf.append(abs(data_mf_red['rf'][trial][1]))

        

#     # print('skipped trials : {}'.format(skipped))
#     last_o_pred=[0]+last_o_pred
#     last_o_pred=last_o_pred[:-1]
#     last_o_pred_other=[0]+last_o_pred_other
#     last_o_pred_other=last_o_pred_other[:-1]


#     # print(last_o_pred_other)
#     # print('# danger : {}'.format(np.sum(pred_rf)))

#     last_o_rew=[]
#     safe_rf=[]
#     last_o_rew_other=[]
#     learning_safe=[]
#     for trial in range(start,len(data_mf)):
#         chosen_state=data_mf_red['choices'][trial]
#         other_state=data_mf_red['other_choices'][trial]
        
#         last_o_rew_other.append(data_mf_red['safes'][trial][other_state-1])
#         last_o_rew.append(data_mf_red['safes'][trial][chosen_state-1])
#         safe_rf.append(abs(data_mf_red['rf'][trial][0]))

#     last_o_rew=[0]+last_o_rew
#     last_o_rew=last_o_rew[:-1]
#     last_o_rew_other=[0]+last_o_rew_other
#     last_o_rew_other=last_o_rew_other[:-1]
#     last_o_saferf=[0]+safe_rf
#     last_o_saferf=last_o_saferf[:-1]
#     last_o_predrf=[0]+pred_rf
#     last_o_predrf=last_o_predrf[:-1]
  



#     X_vars['intercept']=np.array(intercept)
#     X_vars['pred_rf']=np.array(pred_rf) # -1 if safe
#     X_vars['last_pred_rf']=np.array(last_o_predrf)
#     X_vars['last_safe_rf']=np.array(last_o_saferf)
#     X_vars['last_other_rew']=np.array(last_o_rew_other)
#     X_vars['last_other_pred']=np.array(last_o_pred_other)
#     X_vars['last_pred']=np.array(last_o_pred) # -1 if safe
#     X_vars['last_rew']=np.array(last_o_rew) # -1 if safe

#     Y_vars=data_mf_red['switching']

#     c_sub+=1

#     num_PP=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
#     PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0) )].sum()/num_PP

#     num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
#     P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF

#     num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
#     NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
#     num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
#     NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_NP_NRF

 

#     pps.append(PP)
#     p_nrfs.append(P_NRF)
#     np_ps.append(NP_P)
#     np_nrfs.append(NP_NRF)

#     sub_num+=1
      
#     num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0) ).sum()
#     PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_PP

#     num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
#     P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF

#     num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
#     NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
#     num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
#     NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_NP_NRF

#     ppsP.append(PP)
#     p_nrfsP.append(P_NRF)
#     np_psP.append(NP_P)
#     np_nrfsP.append(NP_NRF)

#     #Current REWARD

#     num_PP=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
#     PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_PP

#     num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
#     P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_P_NRF

#     num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
#     NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
#     num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
#     NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_NP_NRF
 
#     Bpps.append(PP)
#     Bp_nrfs.append(P_NRF)
#     Bnp_ps.append(NP_P)
#     Bnp_nrfs.append(NP_NRF)



  
#     num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
#     PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_PP

#     num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
#     P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_P_NRF

#     num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
#     NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
#     num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
#     NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_NP_NRF


#     BppsP.append(PP)
#     Bp_nrfsP.append(P_NRF)
#     Bnp_psP.append(NP_P)
#     Bnp_nrfsP.append(NP_NRF)

    

    
    

# pps=[0 if str(x) == 'nan' else x for x in pps]
# p_nrfs=[0 if str(x) == 'nan' else x for x in p_nrfs]
# np_ps=[0 if str(x) == 'nan' else x for x in np_ps]
# np_nrfs=[0 if str(x) == 'nan' else x for x in np_nrfs]

# ppsP=[0 if str(x) == 'nan' else x for x in ppsP]
# p_nrfsP=[0 if str(x) == 'nan' else x for x in p_nrfsP]
# np_psP=[0 if str(x) == 'nan' else x for x in np_psP]
# np_nrfsP=[0 if str(x) == 'nan' else x for x in np_nrfsP]

# Bpps=[0 if str(x) == 'nan' else x for x in Bpps]
# Bp_nrfs=[0 if str(x) == 'nan' else x for x in Bp_nrfs]
# Bnp_ps=[0 if str(x) == 'nan' else x for x in Bnp_ps]
# Bnp_nrfs=[0 if str(x) == 'nan' else x for x in Bnp_nrfs]

# BppsP=[0 if str(x) == 'nan' else x for x in BppsP]
# Bp_nrfsP=[0 if str(x) == 'nan' else x for x in Bp_nrfsP]
# Bnp_psP=[0 if str(x) == 'nan' else x for x in Bnp_psP]
# Bnp_nrfsP=[0 if str(x) == 'nan' else x for x in Bnp_nrfsP]

# # df=pd.DataFrame()
# # df=df.reset_index()
# # bs=PPS_NON
# # print(len(bs))
# # bp=PPS_CON
# # print(len(bp))


# # df['num_pp']=bs+bp
# # df['type_rf']=['non_consecutive']*400+['consecutive']*400
# # s1 = df.loc[df['type_rf'] == 'non_consecutive']
# # d1 = df.loc[df['type_rf'] == 'consecutive']

# # sns.distplot(s1[['num_pp']], hist=True )
# # sns.distplot(d1[['num_pp']], hist=True)
# # plt.legend(('non_consecutive', 'consecutive'))
# # plt.show()

# # totals=np.asarray(pps)+np.asarray(p_nrfs)+np.asarray(np_ps)+np.asarray(np_nrfs)
# # for i in range(len(pps)):
# #   average=(pps[i]+p_nrfs[i]+np_ps[i]+np_nrfs[i])/4.0
# #   pps[i]=pps[i]-average
# #   p_nrfs[i]=p_nrfs[i]-average
# #   np_ps[i]=np_ps[i]-average
# #   np_nrfs[i]=np_nrfs[i]-average

# # # percent_switch = [np.mean(pps),np.mean(p_nrfs),np.mean(np_ps),np.mean(np_nrfs)]



# # totals=np.asarray(ppsP)+np.asarray(p_nrfsP)+np.asarray(np_psP)+np.asarray(np_nrfsP)
# # for i in range(len(pps)):
# #   average=(ppsP[i]+p_nrfsP[i]+np_psP[i]+np_nrfsP[i])/4.0
# #   ppsP[i]=ppsP[i]-average
# #   p_nrfs[i]=p_nrfsP[i]-average
# #   np_psP[i]=np_psP[i]-average
# #   np_nrfsP[i]=np_nrfsP[i]-average

# sns.set(style='white', font='Arial', font_scale=0.9, color_codes=True, rc=None)


# mean_both_none=np.mean([Bnp_psP[i]+Bnp_ps[i]+p_nrfsP[i]+p_nrfs[i] for i in range(len(np_ps))])
# mean_both_both=np.mean([Bp_nrfsP[i]+Bp_nrfs[i]+np_psP[i]+np_ps[i] for i in range(len(np_ps))])
# from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
#                               AnnotationBbox)
# from matplotlib.patches import Circle
# from matplotlib.patches import Rectangle
# import matplotlib as mpl

# x_col=[]
# y_col=[]
# trial_type=[]
# alphas=[]
# for i in range(len(pps)):
    
#     x_col.append('\U0001F7E1 \u2B24')
#     y_col.append(np_nrfsP[i])
#     x_col.append('NoPred_NoPred')
#     y_col.append(np_psP[i])
#     x_col.append('Pred_Rew')
#     y_col.append(p_nrfsP[i])
#     x_col.append('Pred_Pred')
#     y_col.append(ppsP[i])
#     trial_type.append('PUN \u2192 PUN')
#     trial_type.append('PUN \u2192 PUN')
#     trial_type.append('PUN \u2192 PUN')
#     trial_type.append('PUN \u2192 PUN')
#     alphas.append(1.0)
#     alphas.append(1.0)
#     alphas.append(1.0)
#     alphas.append(1.0)

#     x_col.append('\U0001F7E1 \u2B24')
#     y_col.append(Bnp_nrfs[i])
#     x_col.append('NoPred_NoPred')
#     y_col.append(Bnp_ps[i])
#     x_col.append('Pred_Rew')
#     y_col.append(Bp_nrfs[i])
#     x_col.append('Pred_Pred')
#     y_col.append(Bpps[i])
#     trial_type.append('RWD \u2192 RWD')
#     trial_type.append('RWD \u2192 RWD')
#     trial_type.append('RWD \u2192 RWD')
#     trial_type.append('RWD \u2192 RWD')
#     alphas.append(0.3)
#     alphas.append(0.3)
#     alphas.append(0.3)
#     alphas.append(0.3)

    

    
    
    

# dataf=pd.DataFrame()
# dataf['trial_types']=x_col
# dataf['percent_switch']=y_col
# dataf['trial_type']=trial_type
# dataf['alphas']=alphas
# ax = sns.barplot(x="trial_types", y="percent_switch",hue="trial_type", data=dataf,edgecolor="black",palette=["yellow","darkkhaki"])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[0:], labels=labels[0:])
# ax.set(xticklabels=[])

# # rect = mpl.patches.Circle(
# #     (0.16, -0.04), radius=0.025, color="gold",edgecolor='k', transform=ax.transAxes,
# #     clip_on=False
# # )
# # ax.add_patch(rect)

# # ax.add_patch(Rectangle((-.22,-0.037),0.16, 0.02,color='gray',
# #                               clip_on=False))

# # ax.add_patch(Rectangle((0.78,-0.037),0.16, 0.02,color='gray',
# #                               clip_on=False))
# # ax.add_patch(Rectangle((1.05,-0.037),0.16, 0.02,color='gray',
# #                               clip_on=False))

# # rect = mpl.patches.Circle(
# #     (0.59, -0.04), radius=0.025, color="gold", edgecolor='k',transform=ax.transAxes,
# #     clip_on=False
# # )
# # ax.add_patch(rect)
# # rect = mpl.patches.Circle(
# #     (0.66, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
# #     clip_on=False
# # )
# # ax.add_patch(rect)
# # ax.add_patch(Rectangle((2.78,-0.037),0.16, 0.02,color='gray',
# #                               clip_on=False))

# # rect = mpl.patches.Circle(
# #     (0.92, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
# #     clip_on=False
# # )
# # ax.add_patch(rect)
# # h=np.arange(4)
# # ypos=[np.mean(np_nrfsP),np.mean(np_psP),np.mean(p_nrfsP),np.mean(ppsP)]
# # plt.bar(h,ypos,color=['dodgerblue','dodgerblue','dodgerblue','dodgerblue'])
# ax.set_ylabel('')    
# ax.set_xlabel('')
# plt.savefig('MBGP_all_same',bbox_inches='tight',dpi=300)
# plt.show()


# x_col=[]
# y_col=[]
# trial_type=[]
# alphas=[]
# for i in range(len(pps)):
    
#     x_col.append('\U0001F7E1 \u2B24')
#     y_col.append(Bnp_nrfsP[i])
#     x_col.append('NoPred_NoPred')
#     y_col.append(Bnp_psP[i])
#     x_col.append('Pred_Rew')
#     y_col.append(Bp_nrfsP[i])
#     x_col.append('Pred_Pred')
#     y_col.append(BppsP[i])
#     trial_type.append('RWD \u2192 PUN')
#     trial_type.append('RWD \u2192 PUN')
#     trial_type.append('RWD \u2192 PUN')
#     trial_type.append('RWD \u2192 PUN')
#     alphas.append(1.0)
#     alphas.append(1.0)
#     alphas.append(1.0)
#     alphas.append(1.0)

    

#     x_col.append('\U0001F7E1 \u2B24')
#     y_col.append(np_nrfs[i])
#     x_col.append('NoPred_NoPred')
#     y_col.append(np_ps[i])
#     x_col.append('Pred_Rew')
#     y_col.append(p_nrfs[i])
#     x_col.append('Pred_Pred')
#     y_col.append(pps[i])
#     trial_type.append('PUN \u2192 RWD')
#     trial_type.append('PUN \u2192 RWD')
#     trial_type.append('PUN \u2192 RWD')
#     trial_type.append('PUN \u2192 RWD')
#     alphas.append(0.3)
#     alphas.append(0.3)
#     alphas.append(0.3)
#     alphas.append(0.3)

    
    
    

# dataf=pd.DataFrame()
# dataf['trial_types']=x_col
# dataf['percent_switch']=y_col
# dataf['trial_type']=trial_type
# dataf['alphas']=alphas
# fig, ax = plt.subplots()
# ax = sns.barplot(x="trial_types", y="percent_switch",hue="trial_type", data=dataf,edgecolor="black",palette=["yellow","darkkhaki"])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[0:], labels=labels[0:])
# ax.set(xticklabels=[])
# rect = mpl.patches.Circle(
#     (0.16, -0.04), radius=0.025, color="gold",edgecolor='k', transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)

# ax.add_patch(Rectangle((-.22,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# ax.add_patch(Rectangle((0.78,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))
# ax.add_patch(Rectangle((1.05,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# rect = mpl.patches.Circle(
#     (0.59, -0.04), radius=0.025, color="gold", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# rect = mpl.patches.Circle(
#     (0.66, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# ax.add_patch(Rectangle((2.78,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# rect = mpl.patches.Circle(
#     (0.92, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# h=np.arange(4)
# ypos=[np.mean(np_nrfsP),np.mean(np_psP),np.mean(p_nrfsP),np.mean(ppsP)]
# plt.bar(h,ypos,color=['dodgerblue','dodgerblue','dodgerblue','dodgerblue'])
# ax.set_ylabel('')    
# ax.set_xlabel('')
# plt.savefig('MBGP_all_switch',bbox_inches='tight',dpi=300)
# plt.show()


df=pd.DataFrame()
num_subjects=200
num_trials=160
lrv=list(np.zeros((num_subjects))+0.25)
lrf=np.zeros((2,num_subjects))

lrf_safe=list(np.zeros((num_subjects))+0.15)

pred_change=list(np.zeros((num_subjects)))
rew_change=list(np.zeros((num_subjects)))

# lrf_mild=list(betarnd(2,5,num_subjects))
lrf_each=list(betarnd(3,5,num_subjects))

lr_cfr=list(np.zeros((num_subjects))+0.15)
lr_cfp=list(np.zeros((num_subjects))+0.15)


betas=np.zeros((2,num_subjects))

beta_safe=list(np.zeros(int(num_subjects/2))+5)+list(np.zeros(int(num_subjects/2))+0)
beta_pred=list(np.zeros(int(num_subjects/2))+5)+list(np.zeros(int(num_subjects/2))+0)
betas[0]=beta_safe
betas[1]=beta_pred
#np.zeros((num_subjects))+3.0 # learn about predator feature

#for beta model, there is only a single learning rate over features
lrf2=np.zeros((2,num_subjects))
lrf2[0]=lrf_safe # learnabout safe feature
lrf2[1]=lrf_safe # learnabout safe feature
# lrf2[2]=lrf_safe # learnabout safe feature



mf_betas=list(np.zeros(int(num_subjects/2))+0)+list(np.zeros(int(num_subjects/2))+5)
# beta_irf=list(gammarnd(0.29,3.92,num_subjects))
rfi_rews=list(np.zeros((num_subjects))+0)
rfi_preds=list(np.zeros((num_subjects))+0)
stick=list(np.zeros((num_subjects))+0)


bandits=np.load('safe_firsta.npy')
# bandits2=np.load('safe_firsta.npy')
# bandits=np.concatenate((bandits,bandits2),axis=2)
# print(bandits.shape)

rf2a=np.load('rf_safe_80trials.npy')
rf2=np.flip(rf2a,1)*-1

rf2a=rf2a/3.0
rf2=rf2/3.0
rf1a=np.load('rf_pred_80trials.npy')
rf1=np.flip(rf1a,1)*-1
rf1a=rf1a/3.0
rf1=rf1/3.0

reward_function_timeseries=np.concatenate((rf1,rf2),axis=0)
reward_function_timeseries2=np.concatenate((rf1a,rf2a),axis=0)
reward_function_timeseries3=np.concatenate((rf2a,rf1a),axis=0)
reward_function_timeseries4=np.concatenate((rf2,rf1),axis=0)
#CREATE SYNTHETIC DATA

all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
all_data4,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries2,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
all_data5,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries3,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
all_data6,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries4,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)

all_data3=all_data3+all_data4+all_data5+all_data6
#run logistic regression models on pilot data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_rel as tt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
from math import isnan


c_sub=0


ppsP=[]
p_nrfsP=[]
np_nrfsP=[]
np_psP=[]
pps=[]
p_nrfs=[]
np_nrfs=[]
np_ps=[]


BppsP=[]
Bp_nrfsP=[]
Bnp_nrfsP=[]
Bnp_psP=[]
Bpps=[]
Bp_nrfs=[]
Bnp_nrfs=[]
Bnp_ps=[]

PPS_NON=[]
PPS_CON=[]

sub_num=1
subject_names=[]
corrs_pp_nrnrf=[]
corrs_npp_nrnrf=[]

current_dataset=all_data3
for data_mf in current_dataset:

    length_run=80 #just one environment
    data_mf=data_mf.reset_index(drop=True)

    order_var=1
    first_var=1
    second_var=-1
    start=0
    stop=160

    


 
    p_values_safe=[]
    p_values_pred=[]
    coef_safe=[]
    coef_pred=[]
    failed=0
    X_vars=pd.DataFrame()

    
    data_mf=data_mf.iloc[start:stop]
    start=0

    data_mf=data_mf.reset_index(drop=True)
    #design matrix multinomial logistic regression

    choices=data_mf['choices']
    # null_indices=np.where(pd.isnull(choices))[0]
    data_mf_red=data_mf

    pred_f=2
    rew_f=1

    X1=pd.DataFrame()
    last_o_pred=[]
    pred_rf=[]
    env=[]
    intercept=[]
    order=[]
    skipped=0
    last_o_pred_other=[]
    for trial in range(start,len(data_mf)):
        order.append(order_var)
        intercept.append(1)
        if trial<80:
            env.append(first_var)
        else:
            env.append(second_var)
        
        chosen_state=data_mf_red['choices'][trial]
        other_state=data_mf_red['other_choices'][trial]

        last_o_pred_other.append(data_mf_red['preds'][trial][other_state-1])
        last_o_pred.append(data_mf_red['preds'][trial][chosen_state-1])
        pred_rf.append(abs(data_mf_red['rf'][trial][1]))

        

    # print('skipped trials : {}'.format(skipped))
    last_o_pred=[0]+last_o_pred
    last_o_pred=last_o_pred[:-1]
    last_o_pred_other=[0]+last_o_pred_other
    last_o_pred_other=last_o_pred_other[:-1]


    # print(last_o_pred_other)
    # print('# danger : {}'.format(np.sum(pred_rf)))

    last_o_rew=[]
    safe_rf=[]
    last_o_rew_other=[]
    learning_safe=[]
    for trial in range(start,len(data_mf)):
        chosen_state=data_mf_red['choices'][trial]
        other_state=data_mf_red['other_choices'][trial]
        
        last_o_rew_other.append(data_mf_red['safes'][trial][other_state-1])
        last_o_rew.append(data_mf_red['safes'][trial][chosen_state-1])
        safe_rf.append(abs(data_mf_red['rf'][trial][0]))

    last_o_rew=[0]+last_o_rew
    last_o_rew=last_o_rew[:-1]
    last_o_rew_other=[0]+last_o_rew_other
    last_o_rew_other=last_o_rew_other[:-1]
    last_o_saferf=[0]+safe_rf
    last_o_saferf=last_o_saferf[:-1]
    last_o_predrf=[0]+pred_rf
    last_o_predrf=last_o_predrf[:-1]
  



    X_vars['intercept']=np.array(intercept)
    X_vars['pred_rf']=np.array(pred_rf) # -1 if safe
    X_vars['last_pred_rf']=np.array(last_o_predrf)
    X_vars['last_safe_rf']=np.array(last_o_saferf)
    X_vars['last_other_rew']=np.array(last_o_rew_other)
    X_vars['last_other_pred']=np.array(last_o_pred_other)
    X_vars['last_pred']=np.array(last_o_pred) # -1 if safe
    X_vars['last_rew']=np.array(last_o_rew) # -1 if safe

    Y_vars=data_mf_red['switching']

    c_sub+=1

    num_PP=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0) )].sum()/num_PP

    num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_NP_NRF

 

    pps.append(PP)
    p_nrfs.append(P_NRF)
    np_ps.append(NP_P)
    np_nrfs.append(NP_NRF)

    sub_num+=1
      
    num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0) ).sum()
    PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_PP

    num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_NP_NRF

    ppsP.append(PP)
    p_nrfsP.append(P_NRF)
    np_psP.append(NP_P)
    np_nrfsP.append(NP_NRF)

    #Current REWARD

    num_PP=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_PP

    num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_NP_NRF
 
    Bpps.append(PP)
    Bp_nrfs.append(P_NRF)
    Bnp_ps.append(NP_P)
    Bnp_nrfs.append(NP_NRF)



  
    num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_PP

    num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_NP_NRF


    BppsP.append(PP)
    Bp_nrfsP.append(P_NRF)
    Bnp_psP.append(NP_P)
    Bnp_nrfsP.append(NP_NRF)

    

    
    

pps=[0 if str(x) == 'nan' else x for x in pps]
p_nrfs=[0 if str(x) == 'nan' else x for x in p_nrfs]
np_ps=[0 if str(x) == 'nan' else x for x in np_ps]
np_nrfs=[0 if str(x) == 'nan' else x for x in np_nrfs]

ppsP=[0 if str(x) == 'nan' else x for x in ppsP]
p_nrfsP=[0 if str(x) == 'nan' else x for x in p_nrfsP]
np_psP=[0 if str(x) == 'nan' else x for x in np_psP]
np_nrfsP=[0 if str(x) == 'nan' else x for x in np_nrfsP]

Bpps=[0 if str(x) == 'nan' else x for x in Bpps]
Bp_nrfs=[0 if str(x) == 'nan' else x for x in Bp_nrfs]
Bnp_ps=[0 if str(x) == 'nan' else x for x in Bnp_ps]
Bnp_nrfs=[0 if str(x) == 'nan' else x for x in Bnp_nrfs]

BppsP=[0 if str(x) == 'nan' else x for x in BppsP]
Bp_nrfsP=[0 if str(x) == 'nan' else x for x in Bp_nrfsP]
Bnp_psP=[0 if str(x) == 'nan' else x for x in Bnp_psP]
Bnp_nrfsP=[0 if str(x) == 'nan' else x for x in Bnp_nrfsP]

# df=pd.DataFrame()
# df=df.reset_index()
# bs=PPS_NON
# print(len(bs))
# bp=PPS_CON
# print(len(bp))


# df['num_pp']=bs+bp
# df['type_rf']=['non_consecutive']*400+['consecutive']*400
# s1 = df.loc[df['type_rf'] == 'non_consecutive']
# d1 = df.loc[df['type_rf'] == 'consecutive']

# sns.distplot(s1[['num_pp']], hist=True )
# sns.distplot(d1[['num_pp']], hist=True)
# plt.legend(('non_consecutive', 'consecutive'))
# plt.show()

# totals=np.asarray(pps)+np.asarray(p_nrfs)+np.asarray(np_ps)+np.asarray(np_nrfs)
# for i in range(len(pps)):
#   average=(pps[i]+p_nrfs[i]+np_ps[i]+np_nrfs[i])/4.0
#   pps[i]=pps[i]-average
#   p_nrfs[i]=p_nrfs[i]-average
#   np_ps[i]=np_ps[i]-average
#   np_nrfs[i]=np_nrfs[i]-average

# # percent_switch = [np.mean(pps),np.mean(p_nrfs),np.mean(np_ps),np.mean(np_nrfs)]



# totals=np.asarray(ppsP)+np.asarray(p_nrfsP)+np.asarray(np_psP)+np.asarray(np_nrfsP)
# for i in range(len(pps)):
#   average=(ppsP[i]+p_nrfsP[i]+np_psP[i]+np_nrfsP[i])/4.0
#   ppsP[i]=ppsP[i]-average
#   p_nrfs[i]=p_nrfsP[i]-average
#   np_psP[i]=np_psP[i]-average
#   np_nrfsP[i]=np_nrfsP[i]-average

sns.set(style='white', font='Arial', font_scale=0.9, color_codes=True, rc=None)


mean_both_none=np.mean([Bnp_psP[i]+Bnp_ps[i]+p_nrfsP[i]+p_nrfs[i] for i in range(len(np_ps))])
mean_both_both=np.mean([Bp_nrfsP[i]+Bp_nrfs[i]+np_psP[i]+np_ps[i] for i in range(len(np_ps))])
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                              AnnotationBbox)
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib as mpl

x_col=[]
y_col=[]
trial_type=[]
alphas=[]
for i in range(len(pps)):
    
    x_col.append('\U0001F7E1 \u2B24')
    y_col.append(np_nrfsP[i])
    x_col.append('NoPred_NoPred')
    y_col.append(np_psP[i])
    x_col.append('Pred_Rew')
    y_col.append(p_nrfsP[i])
    x_col.append('Pred_Pred')
    y_col.append(ppsP[i])
    trial_type.append('PUN \u2192 PUN')
    trial_type.append('PUN \u2192 PUN')
    trial_type.append('PUN \u2192 PUN')
    trial_type.append('PUN \u2192 PUN')
    alphas.append(1.0)
    alphas.append(1.0)
    alphas.append(1.0)
    alphas.append(1.0)

    x_col.append('\U0001F7E1 \u2B24')
    y_col.append(Bnp_nrfs[i])
    x_col.append('NoPred_NoPred')
    y_col.append(Bnp_ps[i])
    x_col.append('Pred_Rew')
    y_col.append(Bp_nrfs[i])
    x_col.append('Pred_Pred')
    y_col.append(Bpps[i])
    trial_type.append('RWD \u2192 RWD')
    trial_type.append('RWD \u2192 RWD')
    trial_type.append('RWD \u2192 RWD')
    trial_type.append('RWD \u2192 RWD')
    alphas.append(0.3)
    alphas.append(0.3)
    alphas.append(0.3)
    alphas.append(0.3)

    

    
    
    

dataf=pd.DataFrame()
dataf['trial_types']=x_col
dataf['percent_switch']=y_col
dataf['trial_type']=trial_type
dataf['alphas']=alphas
ax = sns.barplot(x="trial_types", y="percent_switch",hue="trial_type", data=dataf,edgecolor="black",palette=["violet","purple"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
ax.set(xticklabels=[])

# rect = mpl.patches.Circle(
#     (0.16, -0.04), radius=0.025, color="gold",edgecolor='k', transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)

# ax.add_patch(Rectangle((-.22,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# ax.add_patch(Rectangle((0.78,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))
# ax.add_patch(Rectangle((1.05,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# rect = mpl.patches.Circle(
#     (0.59, -0.04), radius=0.025, color="gold", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# rect = mpl.patches.Circle(
#     (0.66, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# ax.add_patch(Rectangle((2.78,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# rect = mpl.patches.Circle(
#     (0.92, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# h=np.arange(4)
# ypos=[np.mean(np_nrfsP),np.mean(np_psP),np.mean(p_nrfsP),np.mean(ppsP)]
# plt.bar(h,ypos,color=['dodgerblue','dodgerblue','dodgerblue','dodgerblue'])
ax.set_ylabel('')    
ax.set_xlabel('')
plt.savefig('MBMF_all_same',bbox_inches='tight',dpi=300)
plt.show()


x_col=[]
y_col=[]
trial_type=[]
alphas=[]
for i in range(len(pps)):
    
    x_col.append('\U0001F7E1 \u2B24')
    y_col.append(Bnp_nrfsP[i])
    x_col.append('NoPred_NoPred')
    y_col.append(Bnp_psP[i])
    x_col.append('Pred_Rew')
    y_col.append(Bp_nrfsP[i])
    x_col.append('Pred_Pred')
    y_col.append(BppsP[i])
    trial_type.append('RWD \u2192 PUN')
    trial_type.append('RWD \u2192 PUN')
    trial_type.append('RWD \u2192 PUN')
    trial_type.append('RWD \u2192 PUN')
    alphas.append(1.0)
    alphas.append(1.0)
    alphas.append(1.0)
    alphas.append(1.0)

    

    x_col.append('\U0001F7E1 \u2B24')
    y_col.append(np_nrfs[i])
    x_col.append('NoPred_NoPred')
    y_col.append(np_ps[i])
    x_col.append('Pred_Rew')
    y_col.append(p_nrfs[i])
    x_col.append('Pred_Pred')
    y_col.append(pps[i])
    trial_type.append('PUN \u2192 RWD')
    trial_type.append('PUN \u2192 RWD')
    trial_type.append('PUN \u2192 RWD')
    trial_type.append('PUN \u2192 RWD')
    alphas.append(0.3)
    alphas.append(0.3)
    alphas.append(0.3)
    alphas.append(0.3)

    
    
    

dataf=pd.DataFrame()
dataf['trial_types']=x_col
dataf['percent_switch']=y_col
dataf['trial_type']=trial_type
dataf['alphas']=alphas
fig, ax = plt.subplots()
ax = sns.barplot(x="trial_types", y="percent_switch",hue="trial_type", data=dataf,edgecolor="black",palette=["violet","purple"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
ax.set(xticklabels=[])
# rect = mpl.patches.Circle(
#     (0.16, -0.04), radius=0.025, color="gold",edgecolor='k', transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)

# ax.add_patch(Rectangle((-.22,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# ax.add_patch(Rectangle((0.78,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))
# ax.add_patch(Rectangle((1.05,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# rect = mpl.patches.Circle(
#     (0.59, -0.04), radius=0.025, color="gold", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# rect = mpl.patches.Circle(
#     (0.66, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# ax.add_patch(Rectangle((2.78,-0.037),0.16, 0.02,color='gray',
#                               clip_on=False))

# rect = mpl.patches.Circle(
#     (0.92, -0.04), radius=0.025, color="black", edgecolor='k',transform=ax.transAxes,
#     clip_on=False
# )
# ax.add_patch(rect)
# h=np.arange(4)
# ypos=[np.mean(np_nrfsP),np.mean(np_psP),np.mean(p_nrfsP),np.mean(ppsP)]
# plt.bar(h,ypos,color=['dodgerblue','dodgerblue','dodgerblue','dodgerblue'])
ax.set_ylabel('')    
ax.set_xlabel('')
plt.savefig('MBMF_all_switch',bbox_inches='tight',dpi=300)
plt.show()