#run logistic regression models on pilot data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import statsmodels.api as sm
from math import isnan
csv_name=''
#cleaned dataframes for each subject seaparated into the version order (e.g., safe block first)
safe_first=[]
danger_first=[]

csv_name='run3_data_a.csv'
#cleaned dataframes for each subject seaparated into the version order (e.g., safe block first)
safe_first=[]
danger_first=[]
danger_indices=[]
safe_indices=[]

data_mf=pd.read_csv(csv_name)
data_mf =data_mf[data_mf.stimulus.eq('You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.').groupby(data_mf.subjectID).cumsum()]
data_mf =data_mf[data_mf.stimulus != 'You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.']
data_mf =data_mf[data_mf.stimulus != 'Great work. You are a third of the way through the task.']
data_mf =data_mf[data_mf.stimulus != 'Great work. You are two thirds of the way through the task.']
subj_IDs = data_mf['subjectID'].str.split(';\s*', expand=True).stack().unique()
#clean up csv`	
counter=0

for subject in subj_IDs:
	
	subs_c=data_mf["subjectID"]==subject
	dfc=data_mf[subs_c]

	dfc1=dfc.reset_index()


	if dfc1['safe_first'][1]==True:
		sf=1
		safe_indices.append(counter)
	else:
		danger_indices.append(counter)
		sf=0
		# print(counter)
		# print('here')
	exec('sub_3_{}=data_mf[subs_c]'.format(counter))
	exec('sub_3_{}=sub_3_{}.reset_index()'.format(counter,counter))
	temp=data_mf[subs_c]
	print('sub : {} , length without errors : {}'.format(counter,len(temp[temp['chosen_state']!='SLOW'])))
	temp=temp.reset_index()
	switching=[]
	for trial in range(len(temp['chosen_state'])):
		if trial>0:
			if temp['chosen_state'][trial]==temp['chosen_state'][trial-1]:
				switching.append(0)
			else:
				switching.append(1)
		else:
			switching.append(0)
	switching=np.array(switching)
	exec('sub_3_{}["switching"]=switching'.format(counter))
	if sf==1:
		exec('safe_first.append(sub_3_{})'.format(counter))
	else:
		exec('danger_first.append(sub_3_{})'.format(counter))
	exec('sub_3_{}.to_csv("sub_3_{}.csv")'.format(counter,counter))
	counter+=1


all_data=safe_first+danger_first
# for sub in all_data:
# 	print(len(sub))
# print(len(all_data))
# print(safe_indices)

# bad_subs=[60,53]