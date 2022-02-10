# Graph all simulations
import seaborn  as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set(context='notebook', style='white', palette='deep', font='arial', font_scale=1.3, color_codes=True, rc=None)




# punishment only
rfi_pred=[-36.8585, -35.286, -34.753, -34.52025, -34.384, -34.39075, -34.407, -34.49025, -34.593, -34.7315, -34.84175, -34.998, -35.1365, -35.276, -35.45725, -35.64, -35.80675, -35.98325, -36.21425, -36.39375]
rfi_rew=[-38.48175, -37.7185, -37.40475, -37.27825, -37.1845, -37.16375, -37.193, -37.25825, -37.30875, -37.3645, -37.47825, -37.56975, -37.669, -37.771, -37.861, -37.905, -38.0185, -38.10625, -38.2275, -38.3425]
mb_only=[-36.80525, -35.554, -34.99575, -34.7685, -34.65375, -34.6175, -34.65, -34.74525, -34.82525, -34.975, -35.1255, -35.252, -35.40725, -35.5645, -35.72025, -35.856, -36.02175, -36.1815, -36.3285, -36.504]
all_data=mb_only+rfi_pred+rfi_rew
all_data=[(-42.98-i)/-10.89 for i in all_data]
print(len(all_data))
labels=['Model Based']*20+['GI Punishment']*20+['GI Reward']*20
rangex=np.arange(0,1,0.05)
rangex=rangex.tolist()*3
df=pd.DataFrame()

df['learning rate']=rangex
df['punishment avoided']=all_data
df['Algorithm']=labels


sns.lineplot(data=df, x="learning rate", y="punishment avoided",hue='Algorithm')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("punishment_learning_hybrid_v_optimal.png",bbox_inches='tight', dpi=300)
plt.show()

# #net reward 
rfi_pred=[8.338, 8.937, 9.311, 9.4055, 9.34525, 9.23625, 9.094, 8.91825, 8.73725, 8.575, 8.35725, 8.10075, 7.846, 7.65225, 7.41575, 7.1135, 6.86275, 6.61325, 6.3325, 6.0255]
rfi_rew=[8.56875, 9.89875, 10.4125, 10.61275, 10.60625, 10.49925, 10.32525, 10.0865, 9.877, 9.63475, 9.35925, 9.07375, 8.8115, 8.53025, 8.24575, 7.93125, 7.652, 7.386, 7.07525, 6.767]
mb_only=[12.479, 14.097, 14.6975, 15.01525, 15.0785, 14.94125, 14.78425, 14.562, 14.31925, 14.037, 13.8015, 13.52575, 13.2295, 12.902, 12.6205, 12.3055, 12.05175, 11.742, 11.38975, 11.09675]
mf=[-1.477, 1.84375, 3.31525, 3.98825, 4.24825, 4.399, 4.34925, 4.28875, 4.22525, 4.2125, 4.05, 3.90675, 3.705, 3.57375, 3.427, 3.17525, 2.957, 2.62175, 2.302, 1.843]
h=[7.04875, 7.71025, 7.95925, 8.0275, 8.0285, 7.9675, 7.82975, 7.7245, 7.54875, 7.377, 7.24225, 7.0485, 6.822, 6.61325, 6.46125, 6.24125, 6.06, 5.88125, 5.669, 5.471]
g=[-1.416, -1.43775, -1.587, -1.581, -1.41925, -1.438, -1.52275, -1.4265, -1.52675, -1.4565, -1.4665, -1.4395, -1.46825, -1.486, -1.644, -1.4545, -1.3795, -1.535, -1.38825, -1.48675]
all_data=mb_only+rfi_pred+rfi_rew+h+mf+g
print(len(all_data))
labels=['Model Based']*20+['GI Punishment']*20+['GI Reward']*20+['GI Both']*20+['Model Free']*20+['Guessing']*20
rangex=np.arange(0,1,0.05)
rangex=rangex.tolist()*6
df=pd.DataFrame()
print(len(rangex))
df['learning rate']=rangex
df['net reward']=all_data
df['Algorithm']=labels

sns.lineplot(data=df, x="learning rate", y="net reward", hue=labels)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("netreward_sim.png",bbox_inches='tight', dpi=300)
plt.show()

# reward only

rfi_pred=[44.273, 45.242, 45.5985, 45.772, 45.87475, 45.9075, 45.94, 45.8565, 45.845, 45.73525, 45.6745, 45.5865, 45.48675, 45.4305, 45.331, 45.22825, 45.15125, 45.031, 44.9355, 44.85825]
mb_only=[46.36425, 47.61, 48.2115, 48.5045, 48.58175, 48.598, 48.57425, 48.464, 48.4275, 48.24725, 48.1015, 48.00375, 47.857, 47.7435, 47.5965, 47.43925, 47.268, 47.10575, 46.9205, 46.73]
rfi_rew=[46.3075, 47.4965, 48.1735, 48.50675, 48.65075, 48.67175, 48.68025, 48.60425, 48.53925, 48.42975, 48.27075, 48.12725, 47.9595, 47.812, 47.664, 47.51825, 47.32025, 47.138, 46.93025, 46.70575]
all_data=mb_only+rfi_pred+rfi_rew
all_data=[(i-41.47)/(51.96-41.47) for i in all_data]
labels=['Model Based']*20+['GI Punishment']*20+['GI Reward']*20
rangex=np.arange(0,1,0.05)
rangex=rangex.tolist()*3
df=pd.DataFrame()

df['learning rate']=rangex
df['reward earned']=all_data
df['Algorithm']=labels


sns.lineplot(data=df, x="learning rate", y="reward earned",hue='Algorithm')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("reward_learning_hybrid_v_optimal.png",bbox_inches='tight', dpi=300)
plt.show()
# sns.lineplot(data=df, x="learning rate", y="net reward")
# plt.savefig("netreward_sim.png",bbox_inches='tight', dpi=300)
# plt.show()

#high worry and low worry - punishment

# hw=[-38.422, -38.2865, -38.278, -38.353, -38.3815, -38.389, -38.463, -38.5315, -38.5955, -38.6305, -38.6935, -38.7905, -38.8145, -38.836, -38.9105, -38.9655, -39.0325, -39.11, -39.1905, -39.2615]
# hw1=[-38.7515, -38.643, -38.6365, -38.6375, -38.681, -38.679, -38.726, -38.791, -38.8295, -38.861, -38.932, -38.9855, -39.0365, -39.0875, -39.1255, -39.1805, -39.2455, -39.3135, -39.369, -39.4635]
# hw2=[-38.117, -37.912, -37.827, -37.7605, -37.741, -37.792, -37.8775, -37.863, -37.9, -37.9255, -37.9365, -38.014, -38.083, -38.1845, -38.2375, -38.308, -38.3995, -38.4675, -38.5225, -38.594]
# hw3=[-38.2215, -38.0715, -37.9915, -37.963, -38.0, -38.0225, -38.0375, -38.0765, -38.127, -38.1845, -38.238, -38.2925, -38.3075, -38.3855, -38.411, -38.476, -38.4955, -38.564, -38.637, -38.741]

# hw=[(hw[i]+hw1[i]+hw2[i]+hw3[i])/4.0 for i in range(len(hw))]
# print(np.mean(hw))
# #low worry - punishment
# lw=[-38.0945, -37.959, -37.8935, -37.8295, -37.886, -37.8945, -37.907, -37.937, -37.998, -38.019, -38.08, -38.1165, -38.1815, -38.25, -38.31, -38.351, -38.4185, -38.492, -38.5605, -38.5955]
# lw1=[-38.3155, -38.169, -38.092, -38.067, -38.0555, -38.0855, -38.0995, -38.179, -38.17, -38.2085, -38.253, -38.286, -38.302, -38.3485, -38.3995, -38.44, -38.4895, -38.554, -38.621, -38.724]
# lw2=[-38.67, -38.578, -38.544, -38.5765, -38.592, -38.6115, -38.685, -38.7145, -38.751, -38.829, -38.9095, -38.937, -39.0075, -39.1115, -39.1675, -39.234, -39.3255, -39.396, -39.4415, -39.5055] 
# lw3=[-38.849,-38.7305, -38.6985, -38.714, -38.7535, -38.797, -38.852, -38.9085, -38.9575, -39.0055, -39.078, -39.127, -39.157, -39.2335, -39.339, -39.432, -39.496, -39.505, -39.603, -39.6975]
# lw=[(lw[i]+lw1[i]+lw2[i]+lw3[i])/4.0 for i in range(len(lw))]
# print(np.mean(lw))

# all_data=hw+lw

# all_data=[(-42.98-i)/-10.89 for i in all_data]
# print(len(all_data))
# labels=['high worry']*20+['low worry']*20
# print(len(labels))
# rangex=np.arange(0,1,0.05)
# rangex=rangex.tolist()*2
# df=pd.DataFrame()
# print(len(rangex))
# df['learning rate']=rangex
# df['propensity to avoid punishment']=all_data
# df['']=labels


# sns.lineplot(data=df, x="learning rate", y="propensity to avoid punishment", hue="", legend='full')
# plt.plot(0.181,  0.4395,marker='o',markersize=10 ,color='blue')
# plt.plot(0.195, 0.428,marker='o',markersize=10, color='orange')                 

# plt.savefig("high_low_worry_fitted_params_punishmentavoidance.png",bbox_inches='tight', dpi=300)
# plt.show()