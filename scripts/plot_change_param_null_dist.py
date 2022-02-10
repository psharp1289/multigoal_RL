import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



rew_changes_means=np.load('rew_changes_all_2rfi.npy')


temp_f=pd.Series(rew_changes_means[:,0],name='null distribution of rew change')
sns.distplot(temp_f)
plt.show()

