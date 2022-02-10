import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set(style='white', font='arial', font_scale=1.3, rc=None)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)




MB = (49.3899 - 42.667449999999995)/(52.3517- 42.667449999999995)
GP_P=( 47.4056 - 42.667449999999995)/(52.3517- 42.667449999999995)
GP_R= (49.388- 42.667449999999995)/(52.3517- 42.667449999999995)
x_col=[]
y_col=[]

x_col.append('MB')
y_col.append(MB)
x_col.append('GP-P')
y_col.append(GP_P)
x_col.append('GP-R')
y_col.append(GP_R)
# x_col.append('MF')
# y_col.append(MF)

fig, ax = plt.subplots()

dataf=pd.DataFrame()
dataf['Strategy']=x_col
dataf['Reward Earned Above Guessing']=y_col
sns.barplot(x="Strategy", y="Reward Earned Above Guessing", ax=ax, data=dataf).set_title('Reward Earned Simulation')
change_width(ax, .50)
h=np.arange(3)
ypos=[MB, GP_P, GP_R]
plt.bar(h,ypos,color=['dodgerblue','darkorange','darkviolet'],width=0.50)
# plt.title('Reward Earned Simulation')
# plt.xlabel('Strategy')
# plt.ylabel('Reward Earned')
plt.savefig('controllers_sim_reward',bbox_inches='tight',dpi=300)
plt.show()


MB = (13.4972 - 0.02)/(19.5178 - 0.02)
GP_P =(11.60 - 0.02)/(19.5178 - 0.02)
GP_R = ( 11.60 - 0.02)/(19.5178 - 0.02)
GP = (9.59235 - 0.02)/(19.5178 - 0.02)
MF = (6.2719 - 0.02)/(19.5178 - 0.02)

x_col=[]
y_col=[]

x_col.append('MB')
y_col.append(MB)
x_col.append('GP-P')
y_col.append(GP_P)
x_col.append('GP-R')
y_col.append(GP_R)
x_col.append('GP')
y_col.append(GP)
x_col.append('MF')
y_col.append(MF)

   

dataf=pd.DataFrame()
dataf['Strategy']=x_col
dataf['Total Points Above Guessing']=y_col
ax = sns.barplot(x="Strategy", y="Total Points Above Guessing", data=dataf).set_title('Total Points Simulation')

h=np.arange(5)
ypos=[MB, GP_P, GP_R, GP, MF]
plt.bar(h,ypos,color=['dodgerblue','darkorange','darkviolet','mediumseagreen','salmon'])
plt.savefig('controllers_sim_netPoints',bbox_inches='tight',dpi=300)
plt.show()

fig, ax = plt.subplots()

MB = (-42.6287 - -35.83685)/ (-42.6287 - -32.83465)
GP_P =(-42.6287 - -35.83685)/ (-42.6287 - -32.83465)
MF = (-42.6287 - -39.0512)/ (-42.6287 - -32.83465)
GP_R = (-42.6287 - -37.79)/ (-42.6287 - -32.83465)
x_col=[]
y_col=[]

x_col.append('MB')
y_col.append((-42.6287 - -35.83685)/(-42.6287 - -32.83465))
x_col.append('GP-P')
y_col.append((-42.6287 - -35.83685)/(-42.6287 - -32.83465))
x_col.append('GP-R')
y_col.append((-42.6287 - -37.79)/(-42.6287 - -32.83465))

dataf=pd.DataFrame()
dataf['Strategy']=x_col
dataf['Punishment Avoided Above Guessing']=y_col
sns.barplot(x="Strategy", y="Punishment Avoided Above Guessing", ax=ax,data=dataf).set_title('Punishment Avoidance Simulation')
change_width(ax, .50)

h=np.arange(3)
ypos=[MB, GP_P, GP_R]
plt.bar(h,ypos,color=['dodgerblue','darkorange','darkviolet'],width=0.50)
plt.savefig('controllers_sim_punishment_avoidance',bbox_inches='tight',dpi=300)
plt.show()