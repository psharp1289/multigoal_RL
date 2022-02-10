# Graph all simulations
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# load environments
safe_feature_pvalues=np.load('safe_feature_pvalues_SafeEnv.npy')
pred_feature_pvalues=np.load('pred_feature_pvalues_SafeEnv.npy')
# equal_env=np.load('optimize_fl_avgrewards.npy')

safe_beta=0
mild_beta=0
pred_beta=0
safe_betas=[]
mild_betas=[]
pred_betas=[]
safes=[]
dangs=[]
current_i=1
for i in range(1,len(safe_feature_pvalues)+2):
	if i==1:
		exec('df_{}=pd.DataFrame()'.format(current_i))

	
	elif i==65:
		safe_beta=0
		mild_beta=0
		pred_beta=0
		# fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
		# cbar_ax = fig.add_axes([.91, .3, .03, .4])

		# for i, ax in enumerate(axn.flat):
		#     sns.heatmap(df, ax=ax,
		#                 cbar=i == 0,
		#                 vmin=0, vmax=1,
		#                 cbar_ax=None if i else cbar_ax)


		# exec('df1= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(current_i-6))
		# exec('df2= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(current_i-5))
		# exec('df3= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(current_i-4))
		# exec('df4= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(current_i-3))
		# exec('df5= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(current_i-2))
		# exec('df6= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(current_i-1))
	
		fig, axn = plt.subplots(1,4,sharex=True,sharey=True)
		for i, ax in enumerate(axn.flat):
			print(i)
			exec('df_{}= df_{}.pivot("mild_beta", "pred_beta", "safe_pval")'.format(i+1,i+1))

			exec("sns.heatmap(df_{}, ax=ax)".format(i+1))
		

		plt.subplots_adjust(wspace=0.5, hspace=None)
		plt.show()

		# fig, axn = plt.subplots(1,4,sharex=True,sharey=True)
		# for i, ax in enumerate(axn.flat):
		# 	print(i)
		# 	exec('df_{}= df_{}.pivot("mild_beta", "pred_beta", "pred_pval")'.format(i+1,i+1))
		# 	exec("sns.heatmap(df_{}, ax=ax)".format(i+1))
		# 	# ax.tick_params(axis='both', which='both', length=0)
		# 	# ax.set_ylabel('')    
		# 	# ax.set_xlabel('')

		# plt.subplots_adjust(wspace=0.5, hspace=None)
		# plt.show()

	safe_betas.append(safe_beta)
	mild_betas.append(mild_beta)
	pred_betas.append(pred_beta)
	if i<65:
		safes.append(safe_feature_pvalues[i-1])
		dangs.append(pred_feature_pvalues[i-1])

		
	
	
	pred_beta+=1
	if pred_beta==4:
		pred_beta=0
		mild_beta+=1
	if mild_beta==4:
		print('here')
		exec('df_{}["safe_beta"]=safe_betas'.format(current_i))
		exec('df_{}["mild_beta"]=mild_betas'.format(current_i))
		exec('df_{}["pred_beta"]=pred_betas'.format(current_i))
		exec('df_{}["safe_pval"]=safes'.format(current_i))
		exec('df_{}["pred_pval"]=dangs'.format(current_i))
		current_i+=1
		exec('df_{}=pd.DataFrame()'.format(current_i))
		mild_beta=0
		safe_beta+=1
		safe_betas=[]
		mild_betas=[]
		pred_betas=[]
		safes=[]
		equals=[]
		dangs=[]



