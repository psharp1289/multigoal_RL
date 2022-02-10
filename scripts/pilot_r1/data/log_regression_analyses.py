#run logistic regression models on pilot data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import statsmodels.api as sm
from math import isnan
csv_name='run1b_data.csv'
#cleaned dataframes for each subject seaparated into the version order (e.g., safe block first)
safe_first=[]
danger_first=[]

data_mf=pd.read_csv(csv_name)
data_mf =data_mf[data_mf.stimulus.eq('You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.').groupby(data_mf.subjectID).cumsum()]
data_mf =data_mf[data_mf.stimulus != 'You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.']
data_mf =data_mf[data_mf.stimulus != 'Great work. You are a third of the way through the task.']
data_mf =data_mf[data_mf.stimulus != 'Great work. You are two thirds of the way through the task.']
subj_IDs = data_mf['subjectID'].str.split(';\s*', expand=True).stack().unique()
#clean up csv
counter=1
for subject in subj_IDs:
	subs_c=data_mf["subjectID"]==subject
	dfc=data_mf[subs_c]

	dfc1=dfc.reset_index()


	if dfc1['safe_first'][1]==True:
		sf=1
	else:
		sf=0
	exec('sub_df_{}=data_mf[subs_c]'.format(counter))
	exec('sub_df_{}=sub_df_{}.reset_index()'.format(counter,counter))
	temp=data_mf[subs_c]
	temp=temp.reset_index()
	switching=[]
	for trial in range(len(temp['chosen_state'])):
		if trial>0:
			if temp['chosen_state'][trial]==temp['chosen_state'][trial-1]:
				switching.append(0)
			else:
				switching.append(1)
			if temp['']
		else:
			switching.append(0)
	switching=np.array(switching)
	exec('sub_df_{}["switching"]=switching'.format(counter))
	if sf==1:
		exec('safe_first.append(sub_df_{})'.format(counter))
	else:
		exec('danger_first.append(sub_df_{})'.format(counter))
	counter+=1

print(safe_first[0]["switching"])

all_safe_coefs=[]
all_safe_ps=[]
all_pred_coefs=[]
all_pred_ps=[]

p_values_safe=[]
p_values_pred=[]
coef_safe=[]
coef_pred=[]
failed=0
#Predicting switch behaviour from interaction of last predator outcome and current predator reward function
X_vars=pd.DataFrame()
last_o_pred=[]
for data_mf in safe_first:
	print('new')
	for trial in range(0,140):
	   last_o_pred.append(data_mf['chosen_state_f_3_outcome'][trial])
	last_o_pred=[0]+last_o_pred
	last_o_pred=last_o_pred[:-1]
	# print(len(last_o_pred))
	# print(len(predator_rfs))
	X_vars['pred_o']=last_o_pred
	X_vars['predator_rf']=abs(data_mf['f_3_reward']/3.0)[0:140]
	X_vars['interaction']=np.multiply(X_vars['pred_o'],X_vars['predator_rf'])
	X_vars=sm.add_constant(X_vars)

	Y=data_mf['switching'][0:140]
	logit = sm.Logit(Y,X_vars)
	try:
	    result = logit.fit(disp=0)
	    p_values_pred.append(result.params[3])
	except:
	    failed+=1



	X_vars=pd.DataFrame()
	last_o_rew=[]
	for trial in range(0,140):
	    last_o_rew.append(data_mf['chosen_state_f_1_outcome'][trial])
	last_o_rew=[0]+last_o_rew
	last_o_rew=last_o_rew[:-1]
	X_vars['safe']=last_o_rew
	X_vars['safe_rf']=abs(data_mf['f_1_reward']/3.0)[0:140]
	X_vars['interaction']=np.multiply(X_vars['safe'],X_vars['safe_rf'])
	X_vars=sm.add_constant(X_vars)

	Y=data_mf['switching'][0:140]
	logit = sm.Logit(Y,X_vars)
	try:
	    result = logit.fit(disp=0)
	    p_values_safe.append(result.params[3])
	except:
	    failed+=1
	#print(result.summary())

	print('avg. z value safe feature: {}'.format(np.mean(p_values_safe)))
	all_safe_ps.append(np.mean(p_values_safe))
	print('avg. z value pred feature: {}'.format(np.mean(p_values_pred)))
	all_pred_ps.append(np.mean(p_values_pred))
	print('regressions failed: {}'.format(failed))

# np.save('safe_feature_pvalues_SafeEnv',all_safe_ps)
# np.save('pred_feature_pvalues_SafeEnv',all_pred_ps)

stop