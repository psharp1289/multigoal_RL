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
# csv_name='run1_data.csv'
# #cleaned dataframes for each subject seaparated into the version order (e.g., safe block first)
# safe_first=[]
# danger_first=[]

# data_mf=pd.read_csv(csv_name)
# data_mf =data_mf[data_mf.stimulus.eq('You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.').groupby(data_mf.subjectID).cumsum()]
# data_mf =data_mf[data_mf.stimulus != 'You passed the quiz! Great work. The task will take about 30 minutes. Press the button to begin.']
# data_mf =data_mf[data_mf.stimulus != 'Great work. You are a third of the way through the task.']
# data_mf =data_mf[data_mf.stimulus != 'Great work. You are two thirds of the way through the task.']
# subj_IDs = data_mf['subjectID'].str.split(';\s*', expand=True).stack().unique()
# #clean up csv`	
# counter=0


subs=[x for x in os.listdir(os.curdir) if x.startswith('sub_3_')]
# subs.remove('sub_3_169.csv')
counter=0

#removed due to choice bias (didn't select an option many times, or because they missed many answers)
# bad_subs=[
# "sub_2a_9.csv",
# "sub_2a_10.csv",]

# for i in bad_subs:
# 	subs.remove(i)
# print(len(subs))
# print(subs)
# bad_subs=[18,41,99,98,173]
bad_subs=[]
# subs_model=[['5ba42e35984ec30001c6018d'], ['5f0576c2b5b074000a8664cc'], ['5e5e865f024e341b083e0dcd'], ['5eca55ce7b00b50119c64518'], ['5ea34772360fe42be0e09295'], ['5f2d9ef399b5e709d6e09d5d'], ['5f7dc0a9912d5b026fc50346'], ['5b9d57e5737d030001ad2cbf'], ['5f7620b93a2dd30771bb322e'], ['5beb492b5324b10001c26ae8'], ['5efd966a27e5f41e732238f5'], ['5e529e9ad9a7ea2ab55835f2'], ['5c9d05e72b3c77001744e8d2'], ['5abe2d5c1667e40001d8250f'], ['5a048d21df3fa800015c33a1'], ['5be1f3f38a2c8000016246e7'], ['5aa308f0b5e2110001c71eb5'], ['5e931494e2bcb364d069bc8a'], ['5e9627e6434ee20a82a7d296'], ['5900dfba92947200019aaa22'], ['5dc161a2ce6a9d0cf9814d5b'], ['5f78b933f653f2489f86469b'], ['5eaa97d07eda6e01ec108107'], ['5d60e3350393f90001015529'], ['5e3de55071cf2606b45c50a6'], ['5f1475dc10e53a053e1fda1e'], ['5e50ecfc5bac4311db8b381b'], ['5cd2fa354cf69b0017a9b2f7'], ['5f4d4f3a2059689f1a2c33db'], ['5f2c16aabf114905864692f9'], ['5a82da17aa46dd00016b8f3b'], ['5ea000d6e849b0000814e9d5'], ['5b39f4f71d4b680001694617'], ['5ee64d41b067113b09098d5e'], ['5be7752e66e3d1000103957f'], ['5c4602713d08e80001369876'], ['5c7458b0d094c900013c3bf0'], ['5ed92f9ea4b0a90f5929cddb'], ['5c59d7eef685da000181d28c'], ['5e13bbae4acfb9a08f0b94c8'], ['5f33e5bdea0471231d3fa54b'], ['5f3ed1b521b5231766f7fef5'], ['5eb6b8d41cf74d5bbc04cc26'], ['5a9c1988f6dfdd0001ea9933'], ['5f11d481902ff50b23166de1'], ['5b65b2773662a8000158c788'], ['59e5177f16ecc200016fefc1'], ['5ebc4f7f306af20607134cfc'], ['5f7af8bd96edc70107b117aa'], ['5e658181af3bd530ce98d4c5'], ['5a94f57289de8200013ecbc8'], ['5f10c8dee777933f01f3d3d6'], ['5c48df1f2381060001bf9fd1'], ['5f74a934c78e6208615393b7'], ['5ecef7476b4f691684117924'], ['598db6bdcceb0f0001b39763'], ['5f51093faa1c4e2d3b82b0a9'], ['56afc8219bf45c000d00d499'], ['5f3a96015cb42d0cf9901e47'], ['5f2335c48a8d300cc838896a'], ['5f3556c7290d724d4ba4a449'], ['55cf6a7b34e9060005e56ca2'], ['5f7631d94ba6c90aec43c7f4'], ['5f6c6f566df913000af123a8'], ['5ea1637e050d8e0008afdff7'], ['5d6d6c5105e693001aa1461b'], ['5d3dd59e791fe2001673f587'], ['5ead973f1e0b1306ab65cf4e'], ['5b07478c6f73510001fdbbda'], ['5ee8f53008d60f24278b816e'], ['5f2a531e8a626e22c1f1d9ad'], ['5f633d376c60233e5dfe5e47'], ['59d49132078dbe00019511c1'], ['5f2c620abb242611c4607aa1'], ['5f53b73f18bf9a6ed513ca61'], ['59e8f20262637600014a7e75'], ['5f4e117dbfcaa83f1b418976'], ['5f7deaf807753f02f9cc2b2a'], ['5cd6c6fd987ddd00187b79fd'], ['56cf683331a5bc000de1d20e'], ['5af1ce4be1b5b8000148afac'], ['5f2c2b2e4d46190913d07955'], ['5e999e7c295393016b7c2577'], ['57444be32801a2000d7a75e9'], ['59f4847293b40c00016c9c5d'], ['5ef4e343d0a7dd102a3c50b8'], ['5d192aef6be726001a39d2f3'], ['5ea601fe68d1210be8aac48d'], ['5eb7dbeede9f986ef1cff0de'], ['njPfksHuHfOFgIwjeW7ChL4PpjD3'], ['5ec679a7d799ae05aa5e4747'], ['5f846dd4e962ce0bf61b95fb'], ['5f86d7c687b0ad035967469f'], ['5f4d17da030fef97ee2d782e'], ['5ea1ac832ac60d0762c238a7'], ['5cbf223b47becc0013c2e355'], ['5ebb10a81635db0ce24f187a'], ['5f83bd0b44cc145582b0ca01'], ['5f8452200539fa07ac939bb8'], ['5f76f9bc8ed0b91b34f829da'], ['5d8a62e934146e001a8ea16d'], ['5ec6bb20ce3f410dcd418f20'], ['5ecbb58e19906d745d70c399'], ['5ea8872b844c430f7d2be8b5'], ['5cd8638310887400162dafec'], ['588c98cf670d6600012bd774'], ['5f22d73c2c28c101e9541d32'], ['5c617d061cbdd10001407426'], ['5810f69c7d3e6d0001df29a5'], ['5f5c079fe585c33ee69c1263'], ['5e2f5d67028f41527321f838'], ['5cbdf7e839447e00017459f3'], ['5c50b130d5172600019f156f'], ['5f8472e84253bf0c72d83923'], ['5e0cbe8919232d467b08ac4c'], ['5c506220ba50dc0001862088'], ['5f400938427f570ee6781225'], ['5de54842a7cb240cb3c34933'], ['5ea82e03b5f29722e4b5835e'], ['5f5a6f4dc6a95e1365660a40'], ['5f343b0f19d95a30e188ee9f'], ['5ed94270ad37064ae646dfd7'], ['5f73ad2ae06cde12229564f1'], ['5ea06b415bd57c02b89d4511'], ['5d83b4c19358b400013494e8'], ['5e1e319cac30fa19d1e131fe'], ['5dc44b73900e3e31ef1e3294'], ['596116d4d865d10001cd6485'], ['5a51f9a3eedc320001420663'], ['5e9d95850b074007356d3c56'], ['5d21228723c079001923a6ba'], ['5f84562b69c963086047a106'], ['5b0abc9ce9270900013b950d'], ['5f75f7a73bf0b60009f88e38'], ['5d8ceadadad82b0016bac50b'], ['5efcb50255a06b097b8d5d29'], ['5ee61c6204aef93753868325'], ['5f5199d96cfa743e8c9d0541'], ['5f1701dd85408e035eb3bd73'], ['5f0ff8ead672e8219f0f9357'], ['5e152c33814bbab3e0eb7371'], ['5ec6c8ad6f911e0e87e58b1f'], ['5968c58012e7f700013b4acc'], ['5c1cdb060108690001f8fe85'], ['5c34f0afffc19a0001a7da70'], ['58fa36995214e3000193281c'], ['5f7366fe5b5261081e3d6fb8'], ['5dd85601e05cad808fd62f3b'], ['5e75cf7b16fafe138cac64c8'], ['5e58436ebdccf5057ddd9190'], ['5f316e50b2f8590176cb8800'], ['5f0d87c546eefe02a4733c5b'], ['5f4697ecad5ee602362af721'], ['5f85f0cab3f4e20ebf578203'], ['5a6b6b34d5d4cb0001d64b8f'], ['5da730b43fc8890017b9a5bd'], ['5c1cb46aaaf1320001426b9e'], ['5cb213bb822bcd0001fea2ad'], ['5ecf68de4145c639871d90d6'], ['5c3f4ccd410e97000128d1cd'], ['5ef8b4207d1275028bb4f9c7'], ['5f02f1d7afde3746b6ce5117'], ['5f3bd0e50bf2cb92bfc8a141'], ['5c6d0da2d5221b000101da81'], ['582468295f4d1c0001c7c467'], ['57acee52bfa77b0001bf31d4'], ['5d2e5d1c649703001a9c920d'], ['57a77b4a8eccbc0001f39fc2'], ['5f18a5e60ce7b80ab30d90ae'], ['5f0c5aa0fc8e5a5afdc94761'], ['5f0a414ff08fe631710e3ce8'], ['5f462e1e8ee312262611e78b'], ['5ec3b64297476e2171182ed6'], ['5ec57b52306f25563b4a11c5'], ['5ebf08453796e31220358f8b'], ['5d923f41e019d1001c713e2b'], ['5d55029618ca41001c7c3583'], ['5b3a03a71d4b680001694793'], ['5f5a4e1f3778490d3e6d1652'], ['597fc43e413c2300012916f1'], ['5b942c77b2352100016db457'], ['5ed12660a41e2603213e1f87'], ['571cc797e1d2ec0012a689a0'], ['5f6cd782abc13d0e70e67d6f'], ['5ecce8a716aee0016eab83cd'], ['57618e8a25224a0006d3756e'], ['5c3ccb06337ac90001a4a9a0'], ['5f0f57173566650fefe59451'], ['56b4b530b2de2a000632cc5f'], ['5bead3d3feaa4a000164c40c'], ['5ec92a7a5ed12a0367a0f95a'], ['5f1de226b5c0873193a834ed']]
# subs_model=[x[0] for x in subs_model]
# bs=[]
# for i in range(len(subs_model)):
# 	if i in bad_subs:
# 		bs.append(subs_model[i])
# remaining_num=[]
greater_5=0
safe_first_subs=[]
danger_first_subs=[]
all_data=[]
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


	if counter in bad_subs:
		print('bad')
	

	elif subject['version'][4].endswith('b'):
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

trials_analysis=67
variables_design_matrix=9
print('len dang subs :{}'.format(len(danger_first_subs)))
print('len safe subs :{}'.format(len(safe_first_subs)))
#Predicting switch behaviour from interaction of last predator outcome and current predator reward function
X_arr=np.zeros((len(all_data),trials_analysis,variables_design_matrix))
X_arr = np.empty((len(all_data),trials_analysis,variables_design_matrix))

# X_arr[:] = np.NaN
Y_arr=np.empty((len(all_data),trials_analysis))
# Y_arr[:] = np.NaN
# X_arr=[]
# Y_arr=[]
percentages=[]
c_sub=0
pred_learning_safe=[]
rew_learning_safe=[]

raw_pps=[]
raw_pnrf=[]
raw_npp=[]
raw_npnrf=[]

sraw_pps=[]
sraw_pnrf=[]
sraw_npp=[]
sraw_npnrf=[]

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

pps2=[]
p_nrfs2=[]
np_nrfs2=[]
np_ps2=[]
pps3=[]
p_nrfs3=[]
np_nrfs3=[]
np_ps3=[]

sub_num=1
subject_names=[]
corrs_pp_nrnrf=[]
corrs_npp_nrnrf=[]
skipsub=0
ver_b=0
current_dataset=all_data
for data_mf in current_dataset:



	# print('length of data : {}'.format(len(data_mf)))
	length_run=80 #just one environment
	data_mf=data_mf.reset_index(drop=True)
	subject_names.append(data_mf.subjectID[0])
	if data_mf['safe_first'][4]==True:
		order_var=1
		first_var=1
		second_var=-1
		start=0
		stop=160
	else:
		order_var=-1
		first_var= -1 # first environment within subject
		second_var = 1
		start=0
		stop=160
	


	# else:
	# if sub_num==129 or sub_num==194:
	# 	if order_var==1:
	# 		# stop=150
	# 		x=2
	# 	else:
	# 		# print('here')
	# 		# length_run=70
	# 		stop=150



	# stop=len(data_mf['chosen_pos'])
	# if stop==164:
	# 	stop-=4
	p_values_safe=[]
	p_values_pred=[]
	coef_safe=[]
	coef_pred=[]
	failed=0
	

	
	data_mf=data_mf.iloc[start:stop]
	start=1
	errors=data_mf[data_mf.reward_received==-3.0]
	before=0
	for i in errors['Unnamed: 0']:
		if i<80:
			before+=1
	data_mf=data_mf[data_mf.reward_received!=-3.0]
	data_mf=data_mf.reset_index(drop=True)
	# design matrix multinomial logistic regression

	choices=data_mf['chosen_pos']
	# null_indices=np.where(pd.isnull(choices))[0]
	data_mf_red=data_mf
	if data_mf['version'][4].endswith('b'):
		# print('version b for sub : {}'.format(sub_num))
		version_rw=1
		ver_b+=1
		pred_f=1
		rew_f=2

	else:
		version_rw=0
		pred_f=2
		rew_f=1

	X1=pd.DataFrame()

	Y1=data_mf_red['chosen_pos']
	last_o_pred=[]
	last_o_pred2=[]
	last_o_rew=[]
	last_o_rew2=[]
	GP_rew=[]
	GP_pun=[]
	
	pred_rf=[]
	last_predrf=[]
	last_saferf=[]
	sw_rwd=[]
	sw_prd=[]
	consec_pred=[]
	consec_rew=[]
	env=[]
	intercept=[]
	version_rws=[]
	order=[]
	skipped=0
	last_o_pred_other=[]
	learning_pred=[]
	switching=[]
	tt=[]
	safe_rf=[]
	last_o_rew_other=[]
	learning_safe=[]
	reward_rec=[]
	counter_tr=0
	rew2=[]
	rew3=[]
	rew4=[]
	current_trial_goals_pred=0
	switches_both_0=[]
	both_pred=0

	for trial in range(start,len(data_mf)):
		if abs(data_mf_red['f_{}_reward'.format(rew_f)][trial-1])!=abs(data_mf_red['f_{}_reward'.format(rew_f)][trial]):
			# if data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-1]==data_mf_red['chosen_state_f_{}_outcome'.format(rew_f)][trial-1]:
		# 		# if abs(data_mf_red['f_{}_reward'.format(pred_f)][trial-1])==1:
			# 		# 	if abs(data_mf_red['f_{}_reward'.format(pred_f)][trial-2])==1:
			if abs(data_mf_red['f_{}_reward'.format(pred_f)][trial])==1:
				tt.append(1)
			else:
				tt.append(0)
			both_pred+=1
			if both_pred<161:

				if data_mf_red['f_{}_reward'.format(pred_f)][trial]==-1:
					current_trial_goals_pred+=1

				counter_tr+=1

				order.append(order_var)

				intercept.append(1)
				version_rws.append(version_rw)
				if np.isnan(data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-1]):
					switching.append(np.random.randint(2))
				elif np.isnan(data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial]):
					switching.append(np.random.randint(2))
				else:
					if str(data_mf_red['chosen_state'][trial])[0]==str(data_mf_red['chosen_state'][trial-1])[0]:
						switching.append(0)
					else:
						switching.append(1)
				if trial<80-before:
					env.append(first_var)
				else:
					env.append(second_var)
				other_states=['1','2']
				try:
					chosen_state=str(data_mf_red['chosen_state'][trial-1])[0]
					other_states.remove(chosen_state)
				except:
					gg='bad'
					# print('bad chosen state name: {}, trial number: {}'.format(chosen_state,trial))
					# print('na')
				last_o_pred_o=0
				for state in other_states:

					if np.isnan(data_mf_red['s_{}_f_{}_outcome'.format(state,pred_f)][trial-1]):
						last_o_pred_o+=np.random.randint(2)
						print('missing data')
					else:
						if data_mf_red['s_{}_f_{}_outcome'.format(state,pred_f)][trial-1]==1:
							last_o_pred_o=1
						else:
							last_o_pred_o+=0

				if last_o_pred_o == 0:
					last_o_pred_other.append(-1)
				else:
					last_o_pred_other.append(1)

				

				if np.isnan(data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-1]):
					last_o_pred.append(np.random.randint(2))
					print('NO DATA')
					pred_rf.append(np.random.randint(2))
					skipped+=1
				else:
					if data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-1]==0:
						last_o_pred.append(-1)
					else:
						last_o_pred.append(1)
					# last_o_pred2.append(data_mf_red['chosen_state_f_{}_outcome'.format(pred_f)][trial-2])
					if data_mf_red['f_{}_reward'.format(pred_f)][trial]==0:
						pred_rf.append(-1)
					else:
						pred_rf.append(1)

					if data_mf_red['f_{}_reward'.format(pred_f)][trial-1]==0:
						last_predrf.append(-1)
					else:
						last_predrf.append(1)
					if abs(data_mf_red['f_{}_reward'.format(pred_f)][trial])==1:
						if abs(data_mf_red['f_{}_reward'.format(pred_f)][trial-1])==1:				
							consec_pred.append(1)
							sw_rwd.append(0)
							sw_prd.append(0)
							consec_rew.append(0)

						else:
							sw_prd.append(1)
							sw_rwd.append(0)
							consec_pred.append(0)
							consec_rew.append(0)

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
						last_o_rew_o+=np.random.randint(2)
					else:
						if data_mf_red['s_{}_f_{}_outcome'.format(state,rew_f)][trial-1]==1:
							last_o_rew_o=1
						else:
							last_o_rew_o+=0
				if last_o_rew_o == 0:
					last_o_rew_other.append(-1)
				else:
					last_o_rew_other.append(1)



				if np.isnan(data_mf_red['chosen_state_f_{}_outcome'.format(rew_f)][trial-1]):
					last_o_rew.append(np.random.randint(2))
					safe_rf.append(np.random.randint(2))

				else:
					if data_mf_red['chosen_state_f_{}_outcome'.format(rew_f)][trial-1]==0:
						last_o_rew.append(-1)
					else:
						last_o_rew.append(1)

					# last_o_rew2.append(data_mf_red['chosen_state_f_{}_outcome'.format(rew_f)][trial-2])
					if data_mf_red['f_{}_reward'.format(rew_f)][trial]==0:
						safe_rf.append(-1)
					else:
						safe_rf.append(1)

					if data_mf_red['f_{}_reward'.format(rew_f)][trial-1]==0:
						last_saferf.append(-1)
					else:
						last_saferf.append(1)
					if abs(data_mf_red['f_{}_reward'.format(rew_f)][trial])==1:
						if abs(data_mf_red['f_{}_reward'.format(rew_f)][trial-1])==1:

							consec_rew.append(1)
							sw_rwd.append(0)
							sw_prd.append(0)
							consec_pred.append(0)

						else:
							sw_rwd.append(1)
							sw_prd.append(0)
							consec_rew.append(0)
							consec_pred.append(0)
				
				if np.isnan(data_mf_red['reward_received'][trial-1]):
					reward_rec.append(np.random.randint(2))
				else:
					data_mf_red['f_{}_reward'.format(rew_f)][trial-1]
					reward_rec.append(data_mf_red['reward_received'][trial-1])

	print('length trials: {}'.format(len(last_predrf)))
	print('num switch to rwd: {}'.format(np.sum(sw_rwd)))
	print('num switch to pred: {}'.format(np.sum(sw_prd)))

	# if both_pred==48:
	# 	print(c_sub)

	# 	print('both_pred = {}'.format(both_pred))
	mbr=np.multiply(np.array(last_o_rew),np.array(safe_rf))
	mbp=np.multiply(np.array(last_o_pred),np.array(pred_rf))
	for i in range(len(last_o_pred)):
		if last_o_pred[i]==last_o_pred_other[i]:
			GP_pun.append(0)
		elif last_o_pred[i]==1:
			GP_pun.append(1)
		else:
			GP_pun.append(-1)
		if last_o_rew[i]==last_o_rew_other[i]:
			GP_rew.append(0)
		elif last_o_rew[i]==1:
			GP_rew.append(1)
		else:
			GP_rew.append(-1)


	rewmb=[]
	rewmf=[]
	rewmb_cf=[]

	for i in range(len(last_o_rew)):
		if GP_rew[i]==1:
			if safe_rf[i]==1:
				rewmb.append(1)
			else:
				rewmb.append(0)
			
		elif GP_rew[i]==-1:
			if safe_rf[i]==1:
				rewmb.append(-1)
			else:
				rewmb.append(0)
		else:
			rewmb.append(0)
		if last_o_rew[i]==1:

			if last_saferf[i]==1:
				rewmf.append(1)
			else:
				rewmf.append(0)
		else:
			if last_saferf[i]==1:
				rewmf.append(-1)
			else:
				rewmf.append(0)
		
			
		# if last_o_rew_other[i]==1:
		# 	if safe_rf[i]==1:
		# 		rewmb_cf.append(1)
		# 	else:
		# 		rewmb_cf.append(0)
		# else:
		# 	if safe_rf[i]==1:
		# 		rewmb_cf.append(-1)
		# 	else:
		# 		rewmb_cf.append(0)

	punmb=[]
	punmb_cf=[]
	punmf=[]
	for i in range(len(last_o_rew)):
		if GP_pun[i]==1:
			if pred_rf[i]==1:
				punmb.append(1)
			else:
				punmb.append(0)

		elif GP_pun[i]==-1:
			if pred_rf[i]==1:
				punmb.append(-1)
			else:
				punmb.append(0)
		else:
			punmb.append(0)
		if last_o_pred[i]==1:
			if last_predrf[i]==1:
				punmf.append(1)
			else:
				punmf.append(0)
		else:
			if last_predrf[i]==1:
				punmf.append(-1)
			else:
				punmf.append(0)
		
		# if last_o_pred_other[i]==1:
		# 	if pred_rf[i]==1:
		# 		punmb_cf.append(1)
		# 	else:
		# 		punmb_cf.append(0)
		# else:
		# 	if pred_rf[i]==1:
		# 		punmb_cf.append(-1)
		# 	else:
		# 		punmb_cf.append(0)
	last_predrf=[-1 if x==0 else x for x in last_predrf]

	sub_num=[]
	for i in last_predrf:
		sub_num.append(c_sub+1)



	X_vars=pd.DataFrame()

	X_vars['intercept']=np.array(intercept)
	X_vars['last_rew']=np.array(GP_rew)
	X_vars['last_pred']=np.array(GP_pun)
	# X_vars['rewo']=np.array(last_o_rew_other)
	# X_vars['puno']=np.array(last_o_pred_other)
	# X_vars['predrf']=np.array(pred_rf)
	X_vars['last_predrf']=np.array(last_predrf)
	X_vars['mbrew']=np.array(rewmb)
	X_vars['mbpun']=np.array(punmb)# -1 if safe
	X_vars['mfrew']=np.array(rewmf)
	X_vars['mfpun']=np.array(punmf)
	# X_vars['rewmb_cf']=np.array(rewmb_cf)
	# X_vars['punmb_cf']=np.array(punmb_cf)# -1 if safe
	X_vars['sub_num']=sub_num


	from scipy.stats import pearsonr as corr
	
	Y_vars=np.array(switching)

	# print(Y_vars.shape)
	print(X_vars.shape)
	print(c_sub)
	X_np=X_vars.to_numpy()
	missing_rows=trials_analysis-len(X_np)
	annex=np.random.randint(2,size=(missing_rows,variables_design_matrix))

	# X_vars=pd.DataFrame()
	# X_vars['intercept']=np.array(intercept)
	# X_vars['last_goal']=np.array(last_predrf)
	# X_vars['current_goal']=np.array(pred_rf)
	# X_vars['last_pred']=np.array(last_o_pred)# -1 if safe
	# X_vars['last_rew']=np.array(last_o_rew)# -1 if safe
	# X_vars['MBrew']=np.multiply(np.array(last_o_rew),np.array(safe_rf))
	# X_vars['MBrew2']=np.multiply(mbr,np.array(last_saferf))
	# X_vars['MBPred']=np.multiply(np.array(last_o_pred),np.array(pred_rf))
	# X_vars['MBPred2']=np.multiply(mbp,np.array(last_predrf))
	print('')
	# print('impute')
	annex[0:missing_rows,0]=1
	annex[0:missing_rows,1]=np.round(np.mean(GP_rew))
	annex[0:missing_rows,2]=np.round(np.mean(GP_pun))
	# annex[0:missing_rows,3]=np.round(np.mean(last_o_rew_other))
	# annex[0:missing_rows,4]=np.round(np.mean(last_o_pred_other))
	# annex[0:missing_rows,3]=np.round(np.mean(pred_rf))
	annex[0:missing_rows,3]=np.round(np.mean(last_predrf))
	annex[0:missing_rows,4]=np.round(np.mean(rewmb))
	annex[0:missing_rows,5]=np.round(np.mean(punmb))
	annex[0:missing_rows,6]=np.round(np.mean(rewmf))
	annex[0:missing_rows,7]=np.round(np.mean(punmf))
	# annex[0:missing_rows,11]=np.round(np.mean(rewmb_cf))
	# annex[0:missing_rows,12]=np.round(np.mean(punmb_cf))
	annex[0:missing_rows,8]=c_sub+1
	
	print('')

	# annex[0:missing_rows,4]=np.round(np.mean(np.multiply(np.array(last_o_rew),np.array(consec_rew))))

	X_np=np.concatenate((X_np,annex),axis=0)
	Y_np=Y_vars
	annex_switch=np.random.randint(2,size=(missing_rows))

	x=np.zeros(round(missing_rows/2))
	y=np.ones(missing_rows-round(missing_rows/2))
	z=np.concatenate((x,y))
	annex_switch[0:missing_rows]=np.round(np.mean(switching))
	Y_np=np.concatenate((Y_np,annex_switch))
	Y_np=Y_np.reshape(trials_analysis,1)
	print('dimensions Y_np {}'.format(Y_np.shape))
	print('dimensions X_np {}'.format(X_np.shape))
	# print(corr(last_o_pred,last_o_rew))
	# print(corr(np.multiply(np.array(last_o_pred),np.array(pred_rf)),np.multiply(np.array(last_o_rew),np.array(safe_rf))))
	# print(corr(np.multiply(np.array(last_o_pred),np.array(pred_rf)),last_o_pred))

	# print(X_vars)
	# Y_arr[c_sub]=Y_np
	X_arr[c_sub]=X_np
	c_sub+=1
	X_npa=np.hstack((X_np,Y_np))
	print(X_npa.shape)
	if c_sub==1:
		X_v=pd.DataFrame(data=X_npa,index=np.arange(1,trials_analysis+1),columns=['i','r','p','g','mbrew','mbpun','mfrew','mfpun','subject','switching'])
		
	else:
		Y_v=pd.DataFrame(data=X_npa,index=np.arange(1,trials_analysis+1),columns=['i','r','p','g','mbrew','mbpun','mfrew','mfpun','subject','switching'])
		frames = [X_v, Y_v]
		X_v = pd.concat(frames)

# np.save('switches_switchtrials',Y_arr)
# np.save('predictors_switchtrials',X_arr)

# print('this many ver b"s: {}'.format(ver_b))
X_v.to_csv('switches_bambi_full_cleaned_ver2.csv')