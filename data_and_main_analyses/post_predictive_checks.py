## import matplotlib.pyplot as plt
import seaborn as sns
from bayesian_bootstrap.bootstrap import mean, highest_density_interval, central_credible_interval
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import numpy as np

#Extract data

gprew=np.load('gprews_empirical.npy')

gppun=np.load('gppuns_empirical.npy')


mbrew=np.load('mbrews_empirical.npy')


mbpun=np.load('mbpuns_empirical.npy')

mfrew=np.load('mfrews_empirical.npy')


mfpun=np.load('mfpuns_empirical.npy')




# CONTRASTS
gpdiff=[np.abs(gprew[i])-gppun[i] for i in range(len(gprew))]
mbdiff=[np.abs(mbrew[i])-mbpun[i] for i in range(len(mbrew))]
mfdiff=[np.abs(mfrew[i])-mfpun[i] for i in range(len(mbrew))]

gp_maineffect=[np.abs(gprew[i])+gppun[i]/2.0 for i in range(len(gprew))]
mb_maineffect=[np.abs(mbrew[i])+mbpun[i]/2.0 for i in range(len(mbpun))]
mf_maineffect=[np.abs(mfrew[i])+mfpun[i]/2.0 for i in range(len(mbpun))]



#Compute HDIs
l_mbr, r_mbr = highest_density_interval(mbrew,alpha=0.05)
l_mbp, r_mbp = highest_density_interval(mbpun,alpha=0.05)
l_gpr, r_gpr = highest_density_interval(gprew,alpha=0.05)
l_gpp, r_gpp = highest_density_interval(gppun,alpha=0.05)
l_mfr, r_mfr = highest_density_interval(mfrew,alpha=0.05)
l_mfp, r_mfp = highest_density_interval(mfpun,alpha=0.05)
l_dmb, r_dmb = highest_density_interval(mbdiff,alpha=0.05)
l_dgp, r_dgp = highest_density_interval(gpdiff,alpha=0.05)
l_dmf, r_dmf = highest_density_interval(mfdiff,alpha=0.05)

l_mb, r_mb = highest_density_interval(mb_maineffect,alpha=0.05)
l_gp, r_gp = highest_density_interval(gp_maineffect,alpha=0.05)
l_mf, r_mf = highest_density_interval(mf_maineffect,alpha=0.05)

rope=0.002
pylab.rcParams['xtick.major.pad']='-.25'
pylab.rcParams['ytick.major.pad']='-.25'

sns.set(context='notebook', style='white', palette='deep', font='arial', font_scale=1.5, color_codes=True, rc=None)

f, axs = plt.subplots(3, 2, figsize=(6,4), sharex=True)

ax1=sns.distplot(mbrew, hist=False,kde_kws={"shade": True},color="dodgerblue", ax=axs[0, 0])
ax1.set(title='MB RWD',xlabel='')

# x,y = ax1.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MB RWD: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_mbr,r_mbr))
axs[0,0].plot([l_mbr, r_mbr],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[0,0].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[0,0].set_ylabel('')

ax0=sns.distplot(mbpun, hist=False,kde_kws={"shade": True},color="dodgerblue", ax=axs[0,1])
ax0.set(title='MB PUN',xlabel='')

axs[0,1].plot([l_mbp, r_mbp],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[0,1].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[0,1].set_ylabel('')

print('')
z=0
print(len(mbpun))
for i in mbpun:
    if i>0:
        z+=1
print('PD MB pun effect = {}'.format(1-((8000-z)/8000.0)))
print('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MB PUN: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_mbp,r_mbp))



ax0=sns.distplot(gprew, hist=False,kde_kws={"shade": True},color="mediumseagreen", ax=axs[1,0])
ax0.set(title='GP RWD',xlabel='')

axs[1,0].plot([l_gpr, r_gpr],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[1,0].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[1,0].set_yticks([0,5])

print('')
z=0
for i in gprew:
    if i<0:
        z+=1
print('PD gp rwd effect = {}'.format(1-((8000-z)/8000.0)))
print('')


# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of RWD: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_gpr,r_gpr))



ax0=sns.distplot(gppun, hist=False,kde_kws={"shade": True},color="mediumseagreen", ax=axs[1,1])
ax0.set(title='GP PUN',xlabel='')

axs[1,1].plot([l_gpp, r_gpp],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[1,1].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[1,1].set_yticks([0,5])

axs[1,1].set_ylabel('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of GP PUN: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_gpp,r_gpp))




ax0=sns.distplot(mfrew, hist=False,kde_kws={"shade": True},color="salmon", ax=axs[2,0])
ax0.set(title='MF RWD',xlabel='')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MF RWD = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_mfr,r_mfr))
axs[2,0].plot([l_mfr, r_mfr],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[2,0].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
# ax1.set_xlim([-1.15, 0.80])
axs[2,0].set_ylabel('')

ax0=sns.distplot(mfpun, hist=False,kde_kws={"shade": True},color="salmon", ax=axs[2,1])
ax0.set(title='MF PUN',xlabel='')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MF PUN = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_mfp,r_mfp))
axs[2,1].plot([l_mfp, r_mfp],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[2,1].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[2,1].set_ylabel('')



plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.25)
f.text(0.5, -.11, 'Parameter Estimate', ha='center')
# f.text(0.035, 0.5, 'Density', va='center', rotation='vertical')
plt.savefig('logisticregression_switchtrialsfPPC_empirical.png', bbox_inches='tight',  dpi=300)

plt.show()


f, axs = plt.subplots(3, 1, figsize=(4,4), sharex=True)
ax0=sns.distplot(mbdiff, hist=False,kde_kws={"shade": True},color="dodgerblue", ax=axs[0])
ax0.set(title='MB RWD-PUN',xlabel='')

axs[0].plot([l_dmb, r_dmb],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[0].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[0].set_ylabel('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MB DIFF: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_dmb,r_dmb))


ax0=sns.distplot(gpdiff, hist=False,kde_kws={"shade": True},color="mediumseagreen", ax=axs[1])
ax0.set(title='GP RWD-PUN',xlabel='')

axs[1].plot([l_dgp, r_dgp],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[1].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')

print('')
z=0
for i in gpdiff:
    if i<0:
        z+=1
print('PD gp DIFF effect = {}'.format(1-((8000-z)/8000.0)))
print('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of GP DIFF: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_dgp,r_dgp))

axs[1].set_yticks([0,3])

ax0=sns.distplot(mfdiff, hist=False,kde_kws={"shade": True},color="salmon", ax=axs[2])
ax0.set(title='MF RWD-PUN',xlabel='')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of DIFF MF = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_dmf,r_dmf))
axs[2].plot([l_dmf, r_dmf],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[2].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.18)
f.text(0.5, -.12, 'Parameter Estimate', ha='center')
# f.text(0.0, 0.5, 'Density', va='center', rotation='vertical')
axs[2].set_xlim(-0.51,0.81)
axs[2].set_ylabel('')

plt.savefig('logisticregression_switchtrials_diffsPPC_empirical.png', bbox_inches='tight',  dpi=300)
plt.show()


f, axs = plt.subplots(3, 1, figsize=(4,4), sharex=True)
ax0=sns.distplot(mb_maineffect, hist=False,kde_kws={"shade": True},color="dodgerblue", ax=axs[0])
ax0.set(title='MB MAIN EFFECT',xlabel='')

axs[0].plot([l_mb, r_mb],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[0].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[0].set_ylabel('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MB MAIN EFFECT: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_mb,r_mb))

axs[0].set_yticks([0,3])

ax0=sns.distplot(gp_maineffect, hist=False,kde_kws={"shade": True},color="mediumseagreen", ax=axs[1])
ax0.set(title='GP MAIN EFFECT',xlabel='')

axs[1].plot([l_gp, r_gp],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[1].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')

print('')
z=0
for i in gp_maineffect:
    if i<0:
        z+=1
print('PD GP MAIN effect = {}'.format(1-((8000-z)/8000.0)))
print('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of GP MAIN EFFECT: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_gp,r_gp))


ax0=sns.distplot(mf_maineffect, hist=False,kde_kws={"shade": True},color="salmon", ax=axs[2])
ax0.set(title='MF MAIN EFFECT',xlabel='')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)

print('')
z=0
for i in mf_maineffect:
    if i<0:
        z+=1
print('PD MF MAIN effect = {}'.format(1-((8000-z)/8000.0)))
print('')

# x,y = ax0.get_lines()[0].get_data()
# y=list(y)
# x=list(x)
# mode= max(y)
# index_mode=y.index(max(y))
# print('mode of MF MAIN EFFECT: = {}'.format(x[index_mode]))
print('CI: [{},{}]'.format(l_mf,r_mf))


axs[2].plot([l_mf, r_mf],[0,0],linewidth=4.0,label='95% HDI',marker='o',color='y')
axs[2].plot([-rope, rope],[0,0],linewidth=4.0,label='ROPE',marker='o',color='black')
axs[2].set_ylabel('')

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.18)

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.18)
f.text(0.5, -.12, 'Parameter Estimate', ha='center')
# f.text(0.0, 0.5, 'Density', va='center', rotation='vertical')
plt.savefig('logisticregression_switchtrials_maineffects_empirical.png', bbox_inches='tight',  dpi=300)

plt.show()