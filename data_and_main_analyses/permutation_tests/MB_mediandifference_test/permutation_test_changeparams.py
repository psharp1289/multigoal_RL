import random
from scipy import optimize
import numpy as np
import os
from scipy.special import logsumexp
from numpy.random import beta,gamma,chisquare,poisson,uniform,logistic,multinomial,binomial
from numpy.random import normal as norm
import pandas as pd
import itertools
import multiprocessing
import numpy as np
from scipy.optimize import minimize as mins
from itertools import repeat
from multiprocessing import Pool
from google.cloud import storage

print('HERE')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mf_key.json'
print('WORKED!!')


possible_envs=[[1,-1],[-1,1]]
pred_changes_all=[]
pred_change_means=[]
rew_changes_all=[]
rew_changes_means=[]
all_envs=[]


randomized_envs=[]
for i in range(192):
    randomized_envs.append(random.choice(possible_envs))


def feature_learner_lik_beta_counterfactual_two_empirical_2lr_2rfinsensitive(samples,data,rng_samples,subject):
    from scipy.special import logsumexp
    sample_size=len(rng_samples)
    num_choices=2
    last_choice=np.zeros((sample_size,num_choices))
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
    rfi_rews=samples[8][rng_samples]
    rfi_preds=samples[9][rng_samples]
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
    data = data.fillna('yup')
    data = data[data['chosen_state']!='yup']
    data=data.reset_index(drop=True)
    env1=randomized_envs[subject][0]
    env2=randomized_envs[subject][1]
    if env1==1:
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

        rf_insensitive_weights=np.concatenate((rfi_rews.reshape(sample_size,1),-1*rfi_preds.reshape(sample_size,1)),axis=1)
        q_rfi1=np.sum((feature_matrix1*rf_insensitive_weights),axis=1) 
        q_rfi2=np.sum((feature_matrix2*rf_insensitive_weights),axis=1)
        q_rfi=np.concatenate((q_rfi1.reshape(sample_size,1),q_rfi2.reshape(sample_size,1)),axis=1)
        q_integrated=q_sr+(q_values*mf_betas)+q_rfi+(last_choice*beta_last)

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
# In[10]:
import pandas as pd
from itertools import repeat,starmap
import numpy as np
import time
import scipy
from numpy.random import beta,gamma,chisquare,normal,poisson,uniform,logistic,multinomial,binomial




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
from scipy.stats import beta,gamma,norm,poisson,uniform,logistic

def fit_hyperparameters(model):
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
            hyperparameters=beta.fit(parameter_full_sample,floc=0,scale=1)
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

        
        rng=np.arange(0,parameter_sample_size,1)
        
        likelihood=lik_func(samples_a,data,rng,subject)
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
        return_dict_info=[[data.subjectID[10]]]
        #return_dict_info=[[subject]]
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
import os
# with open("dataframes_dangerEnv.bin","rb") as data:
#     all_data_input=pickle.load(data)
gcs = storage.Client()

import gcsfs


# number_subjects=1413
csv_file_names=['sub_3_0.csv', 'sub_3_1.csv', 'sub_3_100.csv', 'sub_3_101.csv', 'sub_3_102.csv', 'sub_3_104.csv', 'sub_3_105.csv', 'sub_3_106.csv', 'sub_3_108.csv', 'sub_3_11.csv', 'sub_3_110.csv', 'sub_3_112.csv', 'sub_3_113.csv', 'sub_3_114.csv', 'sub_3_116.csv', 'sub_3_118.csv', 'sub_3_119.csv', 'sub_3_120.csv', 'sub_3_121.csv', 'sub_3_122.csv', 'sub_3_125.csv', 'sub_3_126.csv', 'sub_3_129.csv', 'sub_3_13.csv', 'sub_3_130.csv', 'sub_3_131.csv', 'sub_3_132.csv', 'sub_3_133.csv', 'sub_3_134.csv', 'sub_3_135.csv', 'sub_3_136.csv', 'sub_3_137.csv', 'sub_3_138.csv', 'sub_3_139.csv', 'sub_3_14.csv', 'sub_3_140.csv', 'sub_3_141.csv', 'sub_3_142.csv', 'sub_3_143.csv', 'sub_3_144.csv', 'sub_3_145.csv', 'sub_3_146.csv', 'sub_3_147.csv', 'sub_3_148.csv', 'sub_3_15.csv', 'sub_3_150.csv', 'sub_3_151.csv', 'sub_3_152.csv', 'sub_3_153.csv', 'sub_3_155.csv', 'sub_3_156.csv', 'sub_3_158.csv', 'sub_3_16.csv', 'sub_3_160.csv', 'sub_3_161.csv', 'sub_3_162.csv', 'sub_3_163.csv', 'sub_3_164.csv', 'sub_3_165.csv', 'sub_3_168.csv', 'sub_3_169.csv', 'sub_3_17.csv', 'sub_3_171.csv', 'sub_3_172.csv', 'sub_3_173.csv', 'sub_3_174.csv', 'sub_3_175.csv', 'sub_3_18.csv', 'sub_3_180.csv', 'sub_3_182.csv', 'sub_3_183.csv', 'sub_3_186.csv', 'sub_3_187.csv', 'sub_3_189.csv', 'sub_3_19.csv', 'sub_3_191.csv', 'sub_3_192.csv', 'sub_3_193.csv', 'sub_3_194.csv', 'sub_3_195.csv', 'sub_3_196.csv', 'sub_3_197.csv', 'sub_3_198.csv', 'sub_3_199.csv', 'sub_3_20.csv', 'sub_3_200.csv', 'sub_3_201.csv', 'sub_3_202.csv', 'sub_3_203.csv', 'sub_3_204.csv', 'sub_3_206.csv', 'sub_3_207.csv', 'sub_3_208.csv', 'sub_3_210.csv', 'sub_3_212.csv', 'sub_3_213.csv', 'sub_3_214.csv', 'sub_3_215.csv', 'sub_3_216.csv', 'sub_3_217.csv', 'sub_3_218.csv', 'sub_3_22.csv', 'sub_3_220.csv', 'sub_3_222.csv', 'sub_3_223.csv', 'sub_3_224.csv', 'sub_3_226.csv', 'sub_3_227.csv', 'sub_3_228.csv', 'sub_3_229.csv', 'sub_3_230.csv', 'sub_3_231.csv', 'sub_3_232.csv', 'sub_3_233.csv', 'sub_3_235.csv', 'sub_3_236.csv', 'sub_3_237.csv', 'sub_3_238.csv', 'sub_3_239.csv', 'sub_3_24.csv', 'sub_3_240.csv', 'sub_3_242.csv', 'sub_3_243.csv', 'sub_3_244.csv', 'sub_3_245.csv', 'sub_3_246.csv', 'sub_3_247.csv', 'sub_3_248.csv', 'sub_3_249.csv', 'sub_3_25.csv', 'sub_3_26.csv', 'sub_3_27.csv', 'sub_3_28.csv', 'sub_3_29.csv', 'sub_3_3.csv', 'sub_3_30.csv', 'sub_3_31.csv', 'sub_3_32.csv', 'sub_3_33.csv', 'sub_3_35.csv', 'sub_3_36.csv', 'sub_3_37.csv', 'sub_3_38.csv', 'sub_3_39.csv', 'sub_3_40.csv', 'sub_3_42.csv', 'sub_3_43.csv', 'sub_3_44.csv', 'sub_3_45.csv', 'sub_3_46.csv', 'sub_3_47.csv', 'sub_3_48.csv', 'sub_3_5.csv', 'sub_3_50.csv', 'sub_3_51.csv', 'sub_3_53.csv', 'sub_3_54.csv', 'sub_3_56.csv', 'sub_3_57.csv', 'sub_3_58.csv', 'sub_3_59.csv', 'sub_3_6.csv', 'sub_3_61.csv', 'sub_3_62.csv', 'sub_3_63.csv', 'sub_3_65.csv', 'sub_3_66.csv', 'sub_3_67.csv', 'sub_3_68.csv', 'sub_3_69.csv', 'sub_3_70.csv', 'sub_3_72.csv', 'sub_3_74.csv', 'sub_3_75.csv', 'sub_3_77.csv', 'sub_3_78.csv', 'sub_3_79.csv', 'sub_3_8.csv', 'sub_3_80.csv', 'sub_3_81.csv', 'sub_3_82.csv', 'sub_3_83.csv', 'sub_3_85.csv', 'sub_3_86.csv', 'sub_3_88.csv', 'sub_3_89.csv', 'sub_3_9.csv', 'sub_3_92.csv', 'sub_3_94.csv', 'sub_3_97.csv', 'sub_3_98.csv', 'sub_3_99.csv']
subj_IDs=[]
for csvf in csv_file_names:
    fs = gcsfs.GCSFileSystem(project='hallowed-byte-303916')
    with fs.open("data_changeparams2/{}".format(csvf)) as f:
        dataf = pd.read_csv(f)
        subj_IDs.append(dataf)

print('made it here')


name_1='feature_learner_2states'
# for i in safes:
#   i=i.reset_index(drop=True)
#   if len(i)<51:
#       print(i.subjectID[0])
#       print('')





#### NUMBER SUBJECT #####

number_subjects=192
#number_subjects=40
# stop



parameter_sample_size=100000

                            #name of parameters and hyperpriors
group_parameters_info_mood={'be_mf':['gamma',[1,1]],
                            'lr_neg':['beta',[1,1]],
                            'safe_b':['gamma',[1,1]],
                            'pred_b':['gamma',[1,1]],
                            'lr_val':['beta',[1,1]],
                            'pre_change':['norm',[0,1]],
                            're_change':['norm',[0,1]],
                            'lr_cf':['beta',[1,1]],
                            'rfi_rew':['gamma',[1,1]],
                            'rfi_pred':['gamma',[1,1]],
                            'stick':['norm',[0,1]],

                            }
                            # 'lr_cfac_p':['beta',[1,1]]} 
likelihood='feature_learner_lik_beta_counterfactual_two_empirical_2lr_2rfinsensitive'
build_model(name_1,likelihood,group_parameters_info_mood,number_subjects,parameter_sample_size)
all_models.append(feature_learner_2states)


# group_parameters_info_m={'be_mf':['gamma',[1,1]],'lr_rew':['beta',[1,1]],
#                             'be_safe':['gamma',[1,1]],
#                             'be_pred':['gamma',[1,1]],'lr_val':['beta',[1,1]],
#                             'lr_cfac_r':['beta',[1,1]]}
#                             # 'lr_pred':['beta',[1,1]]}
#                             # 'lr_cfac_p':['beta',[1,1]]} 
# likelihood='feature_learner_lik_beta_counterfactual_two'
# build_model(name_1,likelihood,group_parameters_info_m,number_subjects,parameter_sample_size)
# all_models.append(feature_learner_2states)
# # # Standard model-free Q-learner



# In[12]:




 # Fit models




#Partition data for parallelization
x=np.arange(number_subjects)
cores_subs=12
bins=np.arange(0,number_subjects-1,number_subjects/cores_subs)
subjects_partitioned=[]
for i in range(1,cores_subs+1):
    subjects_partitioned.append(x[np.digitize(x,bins)==i])

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
current_dataset=subj_IDs

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
        # print('current_hyperparms:')
        # print(parameter_info)
        parameter_names=[x[0] for x in parameter_info]
        parameter_disributions=[x[1] for x in parameter_info]
        parameter_sample_size=model.sample_size
        subjects=list(np.arange(0,number_subjects,1))
       
        lik_func=model.lik_func
        return_dict={}
        inputs=zip(subjects,repeat(parameter_info),repeat(current_dataset),
            repeat(lik_func),repeat(parameter_sample_size),repeat(samples_partitioned),repeat(cores))
        
        if __name__=='__main__':    
                pool = Pool(processes=cores_subs)
                results=pool.starmap(process_subject, inputs)
                pool.close()
                pool.join()
                exec('all_results_{} = [item for item in results]'.format(model.name))
                
        exec('all_results=all_results_{}'.format(model.name))


        #fit new hyperparameters from full posterior
        fit_hyperparameters_parallel(model,parameter_info,all_results,number_subjects)
        num_trials=160
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
        
        # read out latest iteration
        print('{}- iBIC old:{}, new: {}\n'.format(model.name, old_bic, new_bic))
        

print('')
# print('final param info:')
# print(parameter_info)
print('')
pred_changes_all.append(model.pre_change.hyperparameters)
rew_changes_all.append(model.re_change.hyperparameters)
pred_change_means.append(model.pre_change.hyperparameters[0])
rew_changes_means.append(model.re_change.hyperparameters[0])
print('preds change final params = {}'.format(model.pre_change.hyperparameters))
print('rew change final params = {}'.format(model.re_change.hyperparameters))
num_subjects=192
all_change_means=[]
pred_betas=[]
safe_betas=[]
for subject in range(num_subjects):
    pred_betas.append(all_results[subject][7][1])
    safe_betas.append(all_results[subject][8][1])

mean_ch_rew = np.mean(safe_betas)
mean_ch_pred = np.mean(pred_betas)
all_change_means.append(mean_ch_pred)
all_change_means.append(mean_ch_rew)


data=[['mean_ch_pred','mean_ch_rew']]
data.append(all_change_means)
df = pd.DataFrame(data[1:],columns=data[0])
df.to_csv('local_file.csv')
gcs.bucket('output_changeparams2').blob('iter_changeparams_{}.csv'.format(np.random.randint(10000000000))).upload_from_filename('local_file.csv', content_type='text/csv')
