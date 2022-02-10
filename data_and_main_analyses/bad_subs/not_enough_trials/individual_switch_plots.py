 #run logistic regression models on pilot data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_rel as tt

import seaborn as sns
import statsmodels.api as sm
from math import isnan


subs=[x for x in os.listdir(os.curdir) if x.startswith('sub_3') ]

counter=0
remaining_num=[]
greater_5=0
safe_first_subs=[]
danger_first_subs=[]
all_data=[]
total_violations=0
pred_wrong=0
rew_wrong=0
low_learning=0
two_signals=0
bad_sub_nums=[]
counter=0
for csvfile in subs:
	# if counter==8:
	# print(csvfile)
	subject=pd.read_csv(csvfile)
	subject=subject.reset_index(drop=True)
	# if subject.subjectID[0]=='5f2aec347b4ac542bc5f34d6':
	# 	print(csvfile)
	# 	stop
	
	all_data.append(subject)
	# if subject.subjectID[10]=='5f4555bdcad55b1274e7ebe1':
	# 	print(csvfile)
	# 	stop
	if subject['safe_first'][4]==True:
		safe_first_subs.append(subject)
		
	else:
		danger_first_subs.append(subject)
		
	# sns.distplot(subject['chosen_state']).set_title('{} Choices'.format(csvfile))
	# plt.show()
	# # ax = sns.lineplot(x="Unnamed: 0", y="time_elapsed", data=subject)
	# plt.show()

	num_sub=counter+1

	counter+=1

print('len dang subs :{}'.format(len(danger_first_subs)))
print('len safe subs :{}'.format(len(safe_first_subs)))

bad_subs=0
c_sub=0
pred_learning_safe=[]
rew_learning_safe=[]
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

sub_num=0
subject_names=[]
corrs_pp_nrnrf=[]
corrs_npp_nrnrf=[]

current_dataset=all_data
for data_mf in current_dataset:

	# print('SUB NUM: {}'.format(sub_num))
	# print(data_mf.subjectID[1])

	# print('length of data : {}'.format(len(data_mf)))
	length_run=80 #just one environment
	data_mf=data_mf.reset_index(drop=True)
	subject_names.append(data_mf.subjectID[0])


	p_values_safe=[]
	p_values_pred=[]
	coef_safe=[]
	coef_pred=[]
	failed=0
	X_vars=pd.DataFrame()

	start=1
	stop=160
	data_mf=data_mf.iloc[0:stop]
	
	errors=data_mf[data_mf.reward_received==-3.0]
	if len(errors)>15:
		print('\n NUMBER OF ERRORS : {}\n'.format(len(errors)))
		print(subs[sub_num])
	before=0
	for i in errors['Unnamed: 0']:
		if i<80:
			before+=1
	data_mf=data_mf[data_mf.reward_received!=-3.0]
	data_mf=data_mf.reset_index(drop=True)
	#design matrix multinomial logistic regression

	choices=data_mf['chosen_pos']
	# null_indices=np.where(pd.isnull(choices))[0]
	data_mf_red=data_mf

	if data_mf['version'][4].endswith('b'):
		# print('version b for sub : {}'.format(sub_num))
		pred_f=1
		rew_f=2
	else:
		# print('version a for sub : {}'.format(sub_num))
		pred_f=2
		rew_f=1

	X1=pd.DataFrame()
	Y1=data_mf_red['chosen_pos']
	last_o_pred=[]
	pred_rf=[]
	last_pred_rf=[]
	env=[]
	intercept=[]
	skipped=0
	last_o_pred_other=[]
	learning_pred=[]
	for trial in range(start,len(data_mf_red)):
		intercept.append(1)

		other_states=['1','2']
		try:
			chosen_state=str(data_mf_red['chosen_state'][trial-1])[0]
			other_states.remove(chosen_state)
		except:
			gg='bad'
		

		last_o_pred_o=0
		for state in other_states:

			if np.isnan(data_mf_red['s_{}_f_{}_outcome'.format(state,pred_f)][trial-1]):
				last_o_pred_o+=0
				print('missing data')
			else:
				if data_mf_red['s_{}_f_{}_outcome'.format(state,pred_f)][trial-1]==1:
					last_o_pred_o=1
				else:
					last_o_pred_o+=0
		
		last_o_pred_other.append(last_o_pred_o)
	
		if np.isnan(data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-1]):
			last_o_pred.append(0)
			print('NO DATA')
			pred_rf.append(0)
			skipped+=1
		else:
			last_o_pred.append(data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-1])
			pred_rf.append(abs(data_mf_red['f_{}_reward'.format(pred_f)][trial]))
			last_pred_rf.append(abs(data_mf_red['f_{}_reward'.format(pred_f)][trial-1]))


	# print('skipped trials : {}'.format(skipped))


	last_o_rew=[]
	safe_rf=[]
	last_o_rew_other=[]
	learning_safe=[]
	for trial in range(start,len(data_mf_red)):
		other_states=['1','2']
		try:
			chosen_state=str(data_mf_red['chosen_state'][trial-1])[0]
			other_states.remove(chosen_state)
		except:
			gg='bad'
			# print('bad chosen state name: {}, trial number: {}'.format(chosen_state,trial))
			# print('na')
		last_o_rew_o=0
		for state in other_states:
		
			if np.isnan(data_mf_red['s_{}_f_{}_outcome'.format(state,rew_f)][trial-1]):
				last_o_rew_o+=0
			else:
				if data_mf_red['s_{}_f_{}_outcome'.format(state,rew_f)][trial-1]==1:
					last_o_rew_o=1
				else:
					last_o_rew_o+=0

		last_o_rew_other.append(last_o_rew_o)

	
		if np.isnan(data_mf_red['chosen_state_f_{}_outcome'.format(rew_f)][trial-1]):
			last_o_rew.append(0)
			safe_rf.append(0)
		else:
			last_o_rew.append(data_mf_red['chosen_state_f_{}_outcome'.format(rew_f)][trial-1])
			safe_rf.append(abs(data_mf_red['f_{}_reward'.format(rew_f)][trial]))

	# if len(data_mf)<160:
	# 	remaining=160-len(data_mf)
	# 	remaining_num.append(remaining)
	# 	if remaining>10:
	# 		greater_5+=1
	# 		# if remaining >18:
	# 		# 	print(data_mf.subjectID[0])
	# 	last_o_rew=[0]+last_o_rew
	# 	last_o_rew=last_o_rew[:-1]
	# 	last_o_rew_other=[0]+last_o_rew_other
	# 	last_o_rew_other=last_o_rew_other[:-1]

	# 	last_o_saferf=[0]+safe_rf
	# 	last_o_saferf=last_o_saferf[:-1]
	# 	last_o_predrf=[0]+pred_rf
	# 	last_o_predrf=last_o_predrf[:-1]
	# 	skipped=0

	# else:

	skipped=0



	X_vars['intercept']=np.array(intercept)
	X_vars['pred_rf']=np.array(pred_rf) # -1 if safe
	X_vars['last_pred_rf']=np.array(last_pred_rf)
	X_vars['last_other_rew']=np.array(last_o_rew_other)
	X_vars['last_other_pred']=np.array(last_o_pred_other)
	X_vars['last_pred']=np.array(last_o_pred) # -1 if safe
	X_vars['last_rew']=np.array(last_o_rew) # -1 if safe
	Y_vars=data_mf_red['switching'][1:160].reset_index(drop=True)

	fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
	ppsP=[]
	p_nrfsP=[]
	np_psP=[]
	np_nrfsP=[]

	try:
		num_PP=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) ).sum()
		PP=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_pred']==1) )].sum()/num_PP
	except:
		print('num PP : {}'.format(num_PP))
		PP=0
	try:
		num_P_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==1) ).sum()
		P_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==1))].sum()/num_P_NRF
	except:
		print('num PP : {}'.format(num_P_NRF))
		P_NRF=0
	try:
		num_NP_P=((X_vars['pred_rf']==1) &  (X_vars['last_pred']==0) ).sum()
		NP_P=Y_vars[((X_vars['pred_rf']==1) &  (X_vars['last_pred']==0) )].sum()/num_NP_P
	except:
		print('num PP : {}'.format(num_NP_P))
		NP_P=0
	try:
		num_NP_NRF=((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) ).sum()
		NP_NRF=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0))].sum()/num_NP_NRF
	except:
		print('num PP : {}'.format(num_NP_NRF))
		NP_NRF=0

	# PPs=((X_vars['pred_rf']==1) & (X_vars['last_pred']==1))
	# PPs=[int(x) for x in PPs.tolist()]
	# NR_NRFs=((X_vars['pred_rf']==1) & (X_vars['last_rew']==0))
	# NR_NRFs=[int(x) for x in NR_NRFs.tolist()]
	# from scipy.stats import pearsonr as rr 
	# r,p=rr(PPs,NR_NRFs)
	# corrs_pp_nrnrf.append(r)

	# NPPs=((X_vars['pred_rf']==1) & (X_vars['last_pred']==0))
	# NPPs=[int(x) for x in NPPs.tolist()]

	# from scipy.stats import pearsonr as rr 
	# r,p=rr(NPPs,NR_NRFs)
	# corrs_npp_nrnrf.append(r)

	average1=(PP+P_NRF+NP_P+NP_NRF)/4.0
	PP=PP-average1
	NP_P=NP_P-average1
	NP_NRF=NP_NRF-average1
	P_NRF=P_NRF-average1
	# learn_pred=-1*NP_P+PP
	# learn_pred_weighted=learn_pred*(len(data_mf)/80.0)
	# pred_learning_safe.append(learn_pred)

	ppsP.append(PP)
	p_nrfsP.append(P_NRF)
	np_psP.append(NP_P)
	np_nrfsP.append(NP_NRF)

	x_col=[]
	y_col=[]
	for i in range(1):
		x_col.append('NoPred_Rew')
		y_col.append(ppsP[i])
		x_col.append('NoPred_NoRew')
		y_col.append(p_nrfsP[i])
		x_col.append('Pred_Rew')
		y_col.append(np_psP[i])
		x_col.append('Pred_NoRew')
		y_col.append(np_nrfsP[i])

	dataf=pd.DataFrame()
	dataf['trial_types']=x_col
	dataf['percent_switch']=y_col
	sns.barplot(x="trial_types", y="percent_switch", data=dataf, ax=ax1).set_title('predator learning')

	# print('NO pred and NO rf {}'.format(num_NP_NRF))
	# # print('number of switches: {}'.format(Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_pred']==0) )].sum()))
	# print(len(X_vars['pred_rf']))
	# print(len(X_vars['last_rew']))
	# print(((X_vars['pred_rf']==0) & (X_vars['last_rew']==1) ).sum())
	# print(len(Y_vars))
	try:
		num_PP=((X_vars['pred_rf']==0) & (X_vars['last_rew']==1) ).sum()
		PP=Y_vars[((X_vars['pred_rf']==0) & (X_vars['last_rew']==1) )].sum()/num_PP
	except:
		print('num PP : {}'.format(num_PP))
		PP=0
	try:
		num_P_NRF=((X_vars['pred_rf']==1) & (X_vars['last_rew']==1) ).sum()
		P_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_rew']==1))].sum()/num_P_NRF
	except:
		print('num PP : {}'.format(num_P_NRF))
		P_NRF=0
	try:
		num_NP_P=((X_vars['pred_rf']==0) &  (X_vars['last_rew']==0) ).sum()
		NP_P=Y_vars[((X_vars['pred_rf']==0) &  (X_vars['last_rew']==0) )].sum()/num_NP_P
	except:
		print('num PP : {}'.format(num_NP_P))
		NP_P=0
	try:
		num_NP_NRF=((X_vars['pred_rf']==1) & (X_vars['last_rew']==0) ).sum()
		NP_NRF=Y_vars[((X_vars['pred_rf']==1) & (X_vars['last_rew']==0))].sum()/num_NP_NRF
	except:
		print('num PP : {}'.format(num_NP_NRF))
		NP_NRF=0

	# average=(PP+P_NRF+NP_P+NP_NRF)/4.0
	# # PP=PP-average
	# # # if PP>0:
	# # # 	PP=0
	# # NP_P=NP_P-average
	# # if NP_P<0:
	# # 	NP_P=0
	# # NP_NRF=NP_NRF-average
	# # P_NRF=P_NRF-average
	# learn_safe=-1*PP+NP_P
	# learn_safe_weighted=learn_safe*(len(data_mf)/80.0)
	# rew_learning_safe.append(learn_safe)

	pps=[]
	p_nrfs=[]
	np_ps=[]
	np_nrfs=[]

	average2=(PP+P_NRF+NP_P+NP_NRF)/4.0
	PP=PP-average2
	NP_P=NP_P-average2
	NP_NRF=NP_NRF-average2
	P_NRF=P_NRF-average2

	pps.append(PP)
	p_nrfs.append(P_NRF)
	np_ps.append(NP_P)
	np_nrfs.append(NP_NRF)


	x_col=[]
	y_col=[]
	for i in range(1):
		x_col.append('NoPred_Rew')
		y_col.append(pps[i])
		x_col.append('NoPred_NoRew')
		y_col.append(p_nrfs[i])
		x_col.append('Pred_Rew')
		y_col.append(np_ps[i])
		x_col.append('Pred_NoRew')
		y_col.append(np_nrfs[i])
	datag=pd.DataFrame()
	datag['trial_types']=x_col
	datag['percent_switch']=y_col
	sns.barplot(x="trial_types", y="percent_switch", data=datag,ax=ax2).set_title('reward learning')
	
	# plt.show()


	# os.chdir('individual_learning')
	# plt.savefig('learning_sub_{}.png'.format(sub_num, dpi=300))
	# os.chdir('..')

	violations=0
	bad_pattern=0
	borderline=0
	perfect = 0
	margin_error=0.01
	too_small=0
	
	#pred wrong
	if ppsP[0]<0:
		if np_psP[0]>0:
			violations+=1
			bad_pattern=1
			pred_wrong+=1

	#rew wrong, pred minimal
	if pps[0]>0:
		if np_ps[0]<0:
			violations+=1
			bad_pattern=1
			rew_wrong+=1

	#Too small a learning signal

	# too_small=0
	# if ((np.abs(ppsP[0])+np.abs(np_psP[0])+np.abs(pps[0])+np.abs(np_ps[0]))/4.0)<0.02:
	# 	print('here')
	# 	print('NOT ENOUGH LEARNING: {} for file {}'.format(violations,subs[sub_num]))
	# 	violations+=1
	
	if perfect==0 and bad_pattern==0:
		if ppsP[0]<0:
			p1= 0
		else:
			p1=ppsP[0]
		if np_psP[0]>0:
			p2= 0
		else:
			p2=np_psP[0]
		if pps[0]>0:
			r1= 0
		else:
			r1=pps[0]
		if np_ps[0]<0:
			r2= 0
		else:
			r2=np_ps[0]


		if (p1+ -1*p2)/2.0 < margin_error:
			violations+=1
			too_small=1
			print('sub borderline: {} for file {}'.format(violations,subs[sub_num]))
			low_learning+=1

		if (-1*r1+r2)/2.0 < margin_error:
			if too_small==0:
				violations+=1
				low_learning+=1
				print('sub borderline: {} for file {}'.format(violations,subs[sub_num]))
				too_small=1

				
	#both wrong -- treat pred like rew and rew like pred
	if ppsP[0]<0:
		if np_psP[0]>0:
			if pps[0]>0:
				if np_ps[0]<0:
					print('REALLY BAD')
					violations+=10 #both are incorrect, delete


	
	#no clear sign anything was learned
	if bad_pattern==0 and too_small==0:
		if (ppsP[0]<0 or np_psP[0]>0) and (pps[0]>0 or np_ps[0]<0):
			violations+=1
			borderline=1
			two_signals+=1
			print('sub borderline: {} for file {}'.format(violations,subs[sub_num]))


	
	if violations>1:
		print('Number of Violations: {} for file {}'.format(violations,subs[sub_num]))

	# if violations==0:
	# 	print('Number with NO Violations: {} for file {}'.format(violations,subs[sub_num]))

	if violations>0:
		total_violations+=1



	
	sub_num+=1
print(len(subs))
print(total_violations)
print('percent violated: {}'.format(total_violations/len(subs)))

print('subs pred wrong: {}'.format(pred_wrong))
print('subs rew wrong: {}'.format(rew_wrong))
print('subs low learning: {}'.format(low_learning))
print('subs only two signals: {}'.format(two_signals))


	# if second<0:
	# 	total+= 1
	# else:
	# 	total-=1

	# if athird>0:
	# 	total+= 1
	# else:
	# 	total-= 1



	# if total<0:
	# 	bad_subs+=1
		# print('SUB NUM : {} IS BAD'.format(sub_num))
		# print(data_mf)

	# plt.show()
	# sub_num+=1
# print('TOTAL BAD SUBS: {}'.format(bad_subs))		
