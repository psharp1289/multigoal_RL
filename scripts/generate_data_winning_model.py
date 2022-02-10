def simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf,lrv,betas,mf_betas,bandits,rew_func,pred_changer,rew_changer,lr_cfs,rfi_rews,rfi_preds,st):
    from numpy.random import choice
    from random import random
    from random import shuffle
    import scipy.special as sf
    import numpy as np
    import pandas as pd
    import scipy as sp
    
    index=0

    for bandit in bandits:
        exec('global bandit_{}; bandit_{}=bandit'.format(index,index))
        index+=1
    num_bandits=len(bandits) 
    

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

            lrf0=lrf[0][subject]
            lrf1=lrf[1][subject]
            lr_features=np.array((lrf0,lrf1))
            lr_cf=lr_cfs[subject]
            last_beta=st[subject]
            lr_value=lrv[subject]
            
            #collect subject level beta weights
            beta_safe=betas[0][subject]
            beta_pred=betas[1][subject]
            pred_changes=pred_changer[subject]
            rew_changes=rew_changer[subject]

            beta_weights_MB=np.array((beta_safe,beta_pred))
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


            #simulate decisions and outcomes based on feature learner that combines MF, MB, Heuristic and Perseveration action values
            for trial in range(num_trials):
                if trial>=80:
                    beta_weights_MB=np.array((beta_safe-rew_changes,beta_pred-pred_changes))
                    
                else:
                    beta_weights_MB=np.array((beta_safe+rew_changes,beta_pred+pred_changes))

                # participant is told which predator they're facing
                current_reward_function=reward_function_time_series[trial]

                #weighting is a combo of beta weights and reward function
                full_weights=beta_weights_MB*current_reward_function

                #derive MB q-values by multiplying feature vectors by reward function
                q_sr=np.matmul(feature_matrix,full_weights)

                #derive MB q-values by multiplying feature vectors by reward function
                q_irf_weights=np.array((rfi_rew,-1*rfi_pred))
                q_insensitive_RF=np.matmul(feature_matrix,q_irf_weights)

                #integrated q_values from MF ,  MB systems, Heuristic and Perseveration
                q_integrated=q_sr+((beta_mf)*(q_values)) + q_insensitive_RF +((last_beta)*(last_choice))
              

                #determine choice via softmax
                action_values = np.exp((q_integrated)-sf.logsumexp(q_integrated));
                values=action_values.flatten()
                draw = choice(values,1,p=values)[0]
                indices = [i for i, x in enumerate(values) if x == draw]
                current_choice=sample(indices,1)[0]
                
                #did participant switch
                if len(choices)==0:
                    switching.append(0)
                else:
                    if current_choice==choices[trial-1]-1:
                        switching.append(0)
                    else:
                        switching.append(1)
                        
                #save choice   
                choices.append(current_choice+1)
             
                #get current feature probabilities
                
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

                #save reward outcome
                outcomes.append(current_outcome)
                
                # Prediction errors over features
                pe_features = feature_outcomes-feature_matrix # Feature prediction errors

                
                current_decision=current_choice

                #cache last choice 
                last_choice=np.zeros((num_bandits,))
                last_choice[current_decision]=1.0

                other_decision=abs(current_choice-1)
                other_choices.append(other_decision+1)

                #Feature updating
                
                #safe feature
                feature_matrix[current_decision,0]=feature_matrix[current_decision,0]+(lr_features[0]*pe_features[current_decision,0])
                feature_matrix[other_decision,0]=feature_matrix[other_decision,0]+(lr_cf*pe_features[other_decision,0])
                
                #predator feature updating
                feature_matrix[current_decision,1]=feature_matrix[current_decision,1]+(lr_features[1]*pe_features[current_decision,1])
                feature_matrix[other_decision,1]=feature_matrix[other_decision,1]+(lr_cf*pe_features[other_decision,1])

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



import numpy as np
from numpy.random import beta as betarnd
from numpy.random import gamma as gammarnd
from numpy.random import normal as normalrnd
from random import sample
from random import shuffle
import pandas as pd
# #assumes you've generated bandits variable with the block above
df=pd.DataFrame()
num_subjects=200
num_trials=160


#MB Learning rates 
lr_feature=list(betarnd(0.32,1.37,num_subjects))
#for beta model, there is only a single learning rate over features
lrf2=np.zeros((2,num_subjects))
lrf2[0]=lr_feature # learnabout safe feature
lrf2[1]=lr_feature # learnabout safe feature

#counterfactual LR
lr_cf=list(betarnd(.18,1.23,num_subjects))

# change beta parameters
pred_change=list(normalrnd(-.15,1.01,num_subjects))
rew_change=list(normalrnd(0.50,1.10,num_subjects))

#MB Beta weights
betas=np.zeros((2,num_subjects))
beta_safe=list(gammarnd(0.60,6.73,num_subjects))
beta_pred=list(gammarnd(.45,4.2,num_subjects))
betas[0]=beta_safe
betas[1]=beta_pred

#MF System Beta and LR
mf_betas=list(gammarnd(1.10,0.54,num_subjects))
lrv=list(betarnd(0.82,0.49,num_subjects))

#Heuristic Beta Weights
rfi_rews=list(gammarnd(0.43,1.68,num_subjects))
rfi_preds=list(gammarnd(0.60,1.43,num_subjects))

#perseveration
stick=list(normalrnd(0.10,0.54,num_subjects))

#feature probability trajectories
bandits=np.load('safe_firsta.npy')

#reward function trajectories
rf1=np.load('rf_safe_80trials.npy')
rf1=rf1/3.0
rf2=np.load('rf_pred_80trials.npy')
rf2=rf2/3.0
reward_function_timeseries=np.concatenate((rf1,rf2),axis=0)


#CREATE SYNTHETIC DATA
all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cf,rfi_rews,rfi_preds,stick)
