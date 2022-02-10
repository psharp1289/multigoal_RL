import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dfs=[x for x in os.listdir(os.curdir) if x.startswith('gpdiff')]

medians=[]
means=[]
above_thershold=0
print('number iterations: {}'.format(len(dfs)))
less_median=0
for i in dfs:
	df=pd.read_csv(i)
	if df.median_diff[0]>0.215:
		print(i)
		print('median')
		print(df.median_diff[0])
		above_thershold+=1

	medians.append(df.median_diff[0])

print('mean of null distribution of medians: {}'.format(np.mean(medians)))
print('p value of observed median diff = {}'.format(above_thershold/len(dfs)))
temp_f=pd.Series(medians,name='null distribution medians')
sns.distplot(temp_f)
# plt.savefig("medians.png",bbox_inches='tight', dpi=300)
plt.show()
