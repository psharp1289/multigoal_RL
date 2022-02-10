import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(context='notebook', style='white', palette='deep', font='arial', font_scale=1.8, color_codes=True, rc=None)

d=pd.read_csv('reduced_param_recovery_best.csv',index_col=0)

d=round(d,2)
mask = np.triu(np.ones_like(d))


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230,20,as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
hmap=sns.heatmap(d,annot=True,cmap=cmap,center=0,mask=mask).set


plt.title('GP-MB Correlations Empirical Data') # title with fontsize 20
plt.xlabel('Fitted Parameters') # x-axis label with fontsize 15
plt.ylabel('Fitted Parameters') # y-axis label with fontsize 15

plt.savefig("Empirical_Data_GP_correlations.png",bbox_inches='tight', dpi=300)
plt.show()
