import numpy as np
import pandas as pd 
import os
from scipy.stats import spearmanr as corr
# os.chdir('../../feature_experiment/data')
# subs=[pd.read_csv(x) for x in os.listdir(os.curdir) if x.startswith('sub_3_')]
# print(len(subs))
# pfa=0
# pfb=0
# rfa=0
# rfb=0

# for sub in subs:
# 	sub = sub.reset_index(drop=True)
# 	if sub.version[10]=='full_pf_a':
# 		pfa+=1
# 	elif sub.version[10]=='full_pf_b':
# 		pfb+=1
# 	elif sub.version[10]=='full_rf_a':
# 		rfa+=1
# 	elif sub.version[10]=='full_rf_b':
# 		rfb+=1

# print('\nRFB:{}\nRFA:{}\nPFA:{}\nPFB:{}'.format(rfb,rfa,pfa,pfb))



bandits=np.load('safe_firsta.npy')
print(corr(bandits[1][1],bandits[1][0]))
print(corr(bandits[1][0],bandits[0][1]))
print(corr(bandits[1][1],bandits[0][1]))
print(corr(bandits[1][1],bandits[0][0]))
print(corr(bandits[0][0],bandits[1][0]))
print(corr(bandits[0][0],bandits[0][1]))