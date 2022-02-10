import pandas as pd
import numpy as np
import os
# high_mf_subs=['54d215e5fdf99b5b9844dac9','5ea2dd16159f0e2092bc021d', '5e13bbae4acfb9a08f0b94c8', '5f33e5bdea0471231d3fa54b', '5a94f57289de8200013ecbc8', '5f5f70aac2a0f3021e66c22c', '5ee8f53008d60f24278b816e', '5c34cece6288ec00016a5363', '5f53b73f18bf9a6ed513ca61', '5eb7dbeede9f986ef1cff0de', '5eff0b0d1311a5436d0d6810', '5f4d17da030fef97ee2d782e', '5f5c079fe585c33ee69c1263', '5efc862d9cbac6036730c847', '5a51f9a3eedc320001420663', '5d8ceadadad82b0016bac50b', '5f0ff8ead672e8219f0f9357', '5c1cb46aaaf1320001426b9e', '57248f38ce62d100092a7715', '5f1caf3bee498903695ad126', '5f0c5aa0fc8e5a5afdc94761', '5ec57b52306f25563b4a11c5']
high_mf_subs=[]
subs=[x for x in os.listdir(os.curdir) if x.startswith('sub_3_')]
bad_subs=[]
lt=0 #less than cutoff for agent that guesses
for sub in subs:
	s=pd.read_csv(sub)
	s=s.reset_index(drop=True)
	if s.subjectID[10] in high_mf_subs:
		print('MOVED')
		os.rename(sub, "bad_subs/{}".format(sub))
	h=s[s.reward_received!=-3.0]
	errors=s[s.reward_received==-3.0]


	bad_trials=s[s.reward_received==-3.0]
	# print('# bad trials = {}'.format(len(bad_trials)))


	current_sum=h.reward_received.sum()
	# print('current sum = {}'.format(current_sum))

	indices=list(h.index)
	# print(indices)
	try:
		ind=indices[0]
	except:
		ind='full_rf_a'
	# try:
	# 	if h.version[ind]=='full_rf_a':
	# 		if current_sum <-2:
	# 			os.rename(sub, "bad_subs/{}".format(sub))
	# 			bad_subs.append(sub)
	# 			lt+=1
	# 	if h.version[ind]=='full_pf_a':
	# 		if current_sum <-2:
	# 			lt+=1
	# 			os.rename(sub, "bad_subs/{}".format(sub))
	# 			bad_subs.append(sub)
	# 	if h.version[ind]=='full_rf_b':
	# 		if current_sum <-2:
	# 			lt+=1
	# 			os.rename(sub, "bad_subs/{}".format(sub))
	# 			bad_subs.append(sub)
	# 	if h.version[ind]=='full_pf_b':
	# 		if current_sum <-2:
	# 			lt+=1
	# 			os.rename(sub, "bad_subs/{}".format(sub))
	# 			bad_subs.append(sub)
	# except:
	# 	os.rename(sub, "bad_subs/{}".format(sub))
	# 	print('bad sub : {}'.format(sub))

	# greater than 10% skipped trials
	try:
		if len(errors)>16:
			os.rename(sub, "bad_subs/{}".format(sub))
	except:
		j='notfound'
	try:
		if len(h)<141:
			os.rename(sub, "bad_subs/{}".format(sub))
	except:
		j='notfound'

print('this many subjects guessed: {}'.format(lt))
print('bad sub list: {}'.format(bad_subs))