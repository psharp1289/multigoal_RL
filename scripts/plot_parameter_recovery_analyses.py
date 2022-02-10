import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(context='notebook', style='white', palette='deep', font='arial', font_scale=1.3, color_codes=True, rc=None)

d=pd.read_csv('param_recovery_best.csv',index_col=0)

d=round(d,2)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230,20,as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
hmap=sns.heatmap(d,annot=True,cmap=cmap,center=0).set


plt.title('Parameter Recovery', fontsize = 20) # title with fontsize 20
plt.xlabel('Ground Truth Parameters', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Fitted Parameters', fontsize = 15) # y-axis label with fontsize 15

plt.savefig("param_recovery_winning_model_lowertriangle2.png",bbox_inches='tight', dpi=300)
plt.show()
