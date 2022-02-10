import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dfs=[x for x in os.listdir(os.curdir) if x.startswith('iter')]

means=[]
above_thershold=0
print('number iterations: {}'.format(len(dfs)))
for i in dfs:
	df=pd.read_csv(i)
	if df.mean_diff[0]>1.36:
		above_thershold+=1
	means.append(df.mean_diff[0])

print('mean of null distribution: {}'.format(np.mean(means)))
print('p value of observed mean = {}'.format(above_thershold/len(dfs)))
temp_f=pd.Series(means,name='null distribution')
sns.distplot(temp_f)
plt.show()