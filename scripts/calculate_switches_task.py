import numpy as np
switches=0
rf2=np.load('rf_safe_80trials.npy')
rf2=rf2/3.0
rf1=np.load('rf_pred_80trials.npy')
rf1=rf1/3.0
reward_function_timeseries=np.concatenate((rf2,rf1),axis=0)
rf=reward_function_timeseries
both_rew=0
both_pun=0
for i in range(1,len(rf)):
	if rf[i][0]!=rf[i-1][0]:
		# print('{} vs. {}'.format(rf[i],rf[i-1]))
		switches+=1
	elif rf[i][0]==1:
		both_rew+=1
	elif rf[i][0]==0:
		both_pun+=1

print('switches: {}'.format(switches))
print('consec rew: {}'.format(both_rew))
print('consec pun: {}'.format(both_pun))
