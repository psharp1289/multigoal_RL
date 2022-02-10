# Graph all simulations
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


safe_rf=np.load('rf_safe_thresh_short.npy')
safes=[]
ppp=[]
for i in safe_rf:
	if 3 in i:
		safes.append(i)
	elif -3 in i:
		ppp.append(i)
print('number safes in safe env: {}'.format(len(safes)))
print('number preds in safe env: {}'.format(len(ppp)))

dang_rf=np.load('rf_pred_thresh_short.npy')
preds=[]
sss=[]
for i in dang_rf:
	if -3 in i:
		preds.append(i)
	elif 3 in i:
		sss.append(i)
print('number preds in dang env: {}'.format(len(preds)))
print('number preds in dang env: {}'.format(len(sss)))

# load environments
safe_env=np.load('SafeEnv_Order3_flipped.npy')
safe_env=safe_env[0,:,0]

diffs_safe=[]
diffs_dang=[]
for i in range(len(safe_env)-36):
	diffs_safe.append(np.abs(safe_env[i]-safe_env[i+36]))

skips=0
for i in range(len(safe_env)):
	if (i+1)%6==0:
		x='skip'
		skips+=1
	else:
		diffs_dang.append(np.abs(safe_env[i]-safe_env[i+1]))

print('skips: {}'.format(skips))
print('SAFE ENV - mean difference due to safe beta weight = {}'.format(np.mean(diffs_safe)))
print('SAFE ENV - mean difference due to danger beta weight = {}'.format(np.mean(diffs_dang)))

dang_env=np.load('DangEnv_Order2_flipped.npy')
dang_env=dang_env[0,:,0]

diffs_safe=[]
diffs_dang=[]
for i in range(len(dang_env)-36):
	diffs_safe.append(np.abs(dang_env[i]-dang_env[i+36]))

skips=0
for i in range(len(dang_env)):
	if (i+1)%6==0:
		x='skip'
		skips+=1
	else:
		diffs_dang.append(np.abs(dang_env[i]-dang_env[i+1]))

print('skips: {}'.format(skips))
print('DANG ENV -mean difference due to safe beta weight = {}'.format(np.mean(diffs_safe)))
print('DANG ENV -mean difference due to danger beta weight = {}'.format(np.mean(diffs_dang)))


m=np.max(dang_env)
d=list(dang_env)
index_max=d.index(m)
print('index max value dang: {}'.format(index_max))


m=np.max(safe_env)
d=list(safe_env)
index_max=d.index(m)
print('index max value safe: {}'.format(index_max))


# eq=list(x[0])
# max_eq=max(eq)
# index_eq=eq.index(max_eq)
# print('max equal RF: {}, index: {}'.format(max_eq,index_eq))
safe_beta=0
mild_beta=0
pred_beta=0
safe_betas=[]
mild_betas=[]
pred_betas=[]
safes=[]
dangs=[]
current_i=1
for i in range(1,len(dang_env)+2):
	if i==1:
		exec('df_{}=pd.DataFrame()'.format(current_i))


	elif i==217:
		df_c=pd.concat([df_1,df_2,df_3,df_4,df_5,df_6])
		df_c1 = df_c.query("mild_beta == 5")
		print(df_c1)
		g = sns.FacetGrid(df_c1, col="safe_beta", col_wrap=3, height=3)
		g.map(sns.pointplot, "pred_beta", "avg_reward_safe_env", order=[0,1,2,3,4,5], color="0.3", ci=None);
		plt.show()

		#df_c=pd.concat([df_1,df_2,df_3,df_4,df_5,df_6])
		df_c2 = df_c.query("mild_beta == 5")
		g = sns.FacetGrid(df_c2, col="pred_beta", col_wrap=3, height=3)
		g.map(sns.pointplot, "safe_beta", "avg_reward_safe_env", order=[0,1,2,3,4,5], color="r", ci=None);
		plt.show()

		#df_c=pd.concat([df_1,df_2,df_3,df_4,df_5,df_6])
		df_c3 = df_c.query("mild_beta == 5")
		g = sns.FacetGrid(df_c3, col="safe_beta", col_wrap=3, height=3)
		g.map(sns.pointplot, "pred_beta", "avg_reward_dang_env", order=[0,1,2,3,4,5], color="0.3", ci=None);
		plt.show()

		#df_c=pd.concat([df_1,df_2,df_3,df_4,df_5,df_6])
		df_c4 = df_c.query("mild_beta == 5")
		g = sns.FacetGrid(df_c4, col="pred_beta", col_wrap=3, height=3)
		g.map(sns.pointplot, "safe_beta", "avg_reward_dang_env", order=[0,1,2,3,4,5], color="r", ci=None);
		plt.show()

		
		fig, axn = plt.subplots(1,6,sharex=True,sharey=True)
		for i, ax in enumerate(axn.flat):
			print(i)
			exec('df_{}= df_{}.pivot("mild_beta", "pred_beta", "avg_reward_dang_env")'.format(i+1,i+1))
			exec("sns.heatmap(df_{}, ax=ax)".format(i+1))
			# ax.tick_params(axis='both', which='both', length=0)
			# ax.set_ylabel('')    
			# ax.set_xlabel('')

		plt.subplots_adjust(wspace=0.5, hspace=None)
		plt.show()
	
	safe_betas.append(safe_beta)
	mild_betas.append(mild_beta)
	pred_betas.append(pred_beta)
	safes.append(safe_env[i-1])
	dangs.append(dang_env[i-1])
	
	
	
	pred_beta+=1
	if pred_beta==6:
		pred_beta=0
		mild_beta+=1

	if mild_beta==6:

		exec('df_{}["safe_beta"]=safe_betas'.format(current_i))
		exec('df_{}["mild_beta"]=mild_betas'.format(current_i))
		exec('df_{}["pred_beta"]=pred_betas'.format(current_i))
		exec('df_{}["avg_reward_safe_env"]=safes'.format(current_i))
		exec('df_{}["avg_reward_dang_env"]=dangs'.format(current_i))#

		current_i+=1
		exec('df_{}=pd.DataFrame()'.format(current_i))
		mild_beta=0
		safe_beta+=1
		safe_betas=[]
		mild_betas=[]
		pred_betas=[]
		safes=[]
		dangs=[]



