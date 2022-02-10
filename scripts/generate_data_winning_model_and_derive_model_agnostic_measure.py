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
##### GENERATE SYNTHETIC DATA

# #### import pandas as pd
# #assumes you've generated bandits variable with the block above
import numpy as np
from numpy.random import beta as betarnd
from numpy.random import gamma as gammarnd
from numpy.random import normal as normalrnd
from random import sample
from random import shuffle
import pandas as pd
# #assumes you've generated bandits variable with the block above
df=pd.DataFrame()
num_subjects=400
num_trials=160
lrv=list(betarnd(0.82,0.49,num_subjects))
lrf=np.zeros((2,num_subjects))
lrf_safe=list(betarnd(0.32,1.37,num_subjects))
lrf_pos=list(betarnd(0.6,2.0,num_subjects))
lrf_neg=list(betarnd(0.4,1.2,num_subjects))

pred_change=list(normalrnd(-.15,1.01,num_subjects))
rew_change=list(normalrnd(0.50,1.10,num_subjects))

# lrf_mild=list(betarnd(2,5,num_subjects))
lrf_each=list(betarnd(3,5,num_subjects))

lr_cfr=list(betarnd(.18,1.23,num_subjects))
lr_cfp=list(betarnd(0.40,1.2,num_subjects))

lrf[0]=np.zeros((num_subjects))+0.1 # learnabout safe feature
# lrf[1]=lrf_mild # learn about mild feature (small reward)
lrf[1]=np.zeros((num_subjects))+0.1 # learn about predator feature

betas=np.zeros((2,num_subjects))
# beta_safe=list(gammarnd(0.75,6,num_subjects))
# beta_pred=list(gammarnd(0.62,3.71,num_subjects))
beta_safe=list(gammarnd(0.60,6.73,num_subjects))
beta_pred=list(gammarnd(.45,4.2,num_subjects))
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
mf_betas=list(gammarnd(1.10,0.54,num_subjects))
# beta_irf=list(gammarnd(0.29,3.92,num_subjects))
rfi_rews=list(gammarnd(0.43,1.68,num_subjects))
rfi_preds=list(gammarnd(0.60,1.43,num_subjects))
stick=list(normalrnd(0.10,0.54,num_subjects))
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

all_data3,outcomes3,rewards3,choices3,safes3,preds3=simulate_one_step_featurelearner_counterfactual_2states_2iRF_sticky(num_subjects,num_trials,lrf2,lrv,betas,mf_betas,bandits,reward_function_timeseries,pred_change,rew_change,lr_cfr,lr_cfp,rfi_rews,rfi_preds,stick)
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
        pred_rf.append(abs(reward_function_timeseries[trial][1]))

        

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
        safe_rf.append(abs(reward_function_timeseries[trial][0]))

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

 
      
    num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0) )].sum()/num_PP

    num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_NP_NRF

    #Current REWARD
  
    # num_PP=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    # PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0) )].sum()/num_PP

    # num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    # P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    # num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    # NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    # num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1)).sum()
    # NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_NP_NRF
   
    # num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) ).sum()
    # PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1)  )].sum()/num_PP

    # num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) ).sum()
    # P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) )].sum()/num_P_NRF

    # num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) ).sum()
    # NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) )].sum()/num_NP_P
 
    # num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)).sum()
    # NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) )].sum()/num_NP_NRF
  
    # average=(PP+P_NRF+NP_P+NP_NRF)/4.0
    
    
    # PP=PP-average
    # P_NRF=P_NRF-average
    # NP_P=NP_P-average
    # NP_NRF=NP_NRF-average

    ppsP.append(PP)
    p_nrfsP.append(P_NRF)
    np_psP.append(NP_P)
    np_nrfsP.append(NP_NRF)

    num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_PP

    num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    num_NP_P=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    NP_P=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_NP_NRF

    #Current REWARD
    # num_PP=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    # PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_PP

    # num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    # P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_P_NRF

    # num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0)).sum()
    # NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==0))].sum()/num_NP_P
 
    # num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0)& (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1)).sum()
    # NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) & (X_vars['last_pred_rf']==0) & (X_vars['last_rew']==1))].sum()/num_NP_NRF
    # average=(PP+P_NRF+NP_P+NP_NRF)/4.0

    # num_PP=((X_vars['pred_rf']==0) & (X_vars['last_rew']==1) ).sum()
    # PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_rew']==1)  )].sum()/num_PP

    # num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_rew']==1) ).sum()
    # P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_rew']==1) )].sum()/num_P_NRF

    # num_NP_P=((X_vars['pred_rf']==0) & (X_vars['last_rew']==0) ).sum()
    # NP_P=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_rew']==0) )].sum()/num_NP_P
 
    # num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_rew']==0)).sum()
    # NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_rew']==0) )].sum()/num_NP_NRF

    # average=(PP+P_NRF+NP_P+NP_NRF)/4.0
    
    # PP=PP-average
    # P_NRF=P_NRF-average
    # NP_P=NP_P-average
    # NP_NRF=NP_NRF-average

    pps.append(PP)
    p_nrfs.append(P_NRF)
    np_ps.append(NP_P)
    np_nrfs.append(NP_NRF)

    sub_num+=1
    

pps=[0 if str(x) == 'nan' else x for x in pps]
p_nrfs=[0 if str(x) == 'nan' else x for x in p_nrfs]
np_ps=[0 if str(x) == 'nan' else x for x in np_ps]
np_nrfs=[0 if str(x) == 'nan' else x for x in np_nrfs]

ppsP=[0 if str(x) == 'nan' else x for x in ppsP]
p_nrfsP=[0 if str(x) == 'nan' else x for x in p_nrfsP]
np_psP=[0 if str(x) == 'nan' else x for x in np_psP]
np_nrfsP=[0 if str(x) == 'nan' else x for x in np_nrfsP]



    
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




x_col=[]
y_col=[]
for i in range(len(pps)):
    x_col.append('NoPred_Rew')
    y_col.append(np_nrfsP[i])
    x_col.append('NoPred_NoRew')
    y_col.append(np_psP[i])
    x_col.append('Pred_Rew')
    y_col.append(p_nrfsP[i])
    x_col.append('Pred_NoRew')
    y_col.append(ppsP[i])
    

dataf=pd.DataFrame()
dataf['trial_types']=x_col
dataf['percent_switch']=y_col
ax = sns.barplot(x="trial_types", y="percent_switch", data=dataf).set_title('Pred RF Last Pred RF Current')

# plt.savefig('RFs_PredRew_CF_YES.png',dpi=300)
plt.savefig('predatorlast_Predcurrent_2lrmodel_2iRF',dpi=300)

plt.show()



x_col=[]
y_col=[]
for i in range(len(pps)):
    x_col.append('NoPred_Rew')
    y_col.append(np_nrfs[i])
    x_col.append('NoPred_NoRew')
    y_col.append(np_ps[i])
    x_col.append('Pred_Rew')
    y_col.append(p_nrfs[i])
    x_col.append('Pred_NoRew')
    y_col.append(pps[i])
   
   

dataf=pd.DataFrame()
dataf['trial_types']=x_col
dataf['percent_switch']=y_col
ax = sns.barplot(x="trial_types", y="percent_switch", data=dataf).set_title('Rew RF Last Pred RF Current')

plt.savefig('Rewlast_Predcurrent_2lrmodel_2iRF',dpi=300)
plt.show()


x_col=[]
y_col=[]
for i in range(len(pps)):
    x_col.append('Pred_PredRF')
    y_col.append(ppsP[i])
    x_col.append('Pred_RewRF')
    y_col.append(p_nrfsP[i])
    x_col.append('NoPred_PredRF')
    y_col.append(np_psP[i])
    x_col.append('NoPred_RewRF')
    y_col.append(np_nrfsP[i])
    
    
    
    

# dataf=pd.DataFrame()
# dataf['trial_types']=x_col
# dataf['percent_switch']=y_col
# ax = sns.barplot(x="trial_types", y="percent_switch", data=dataf)

# plt.savefig('Predator_Learning_Simulatied_WinningModelRFI',dpi=300)

# plt.show()
