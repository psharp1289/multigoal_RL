import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='white', palette='Greys', font='arial', font_scale=1.5, rc=None)
df=pd.DataFrame()
winning_model=32709.3833984
BICs=[38009.37073804648-winning_model,34676.46504166039-winning_model,34221.15410923616-winning_model,34025.5448863697-winning_model,
33896.481327053836-winning_model, 
32873.48764737718-winning_model,32817.2429937819-winning_model,32796.49376811623-winning_model,32754.32466701125-winning_model,
33098.176577920036-winning_model, 32810.86003185599-winning_model,33105.92139841622-winning_model, 
32868.395910117106-winning_model,32925.03026089108-winning_model,
33437.36913228646-winning_model,34474.5962848594-winning_model, 32825.15011052856-winning_model]


def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQʀₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)
  
# display subscript
MB_RR=('MB{}{}'.format(get_sub('R'),get_sub('R')))
# display subscript
GP_RR=('GP{}{}'.format(get_sub('R'),get_sub('R')))



df['Change in iBIC from Winning Model: \n {}+MF+GP'.format(MB_RR)]=BICs
df['Model Name'] = ['Null (AP)','MB no CF', 'MB 1LR','MB', MB_RR,
'MB+MF+{}'.format(GP_RR),'{}+MF+{}'.format(MB_RR,GP_RR),'MB+RFHist+MF+GP','{}+MF+GP1'.format(MB_RR),
'MBLR+MF+GPLR','MB+MF+GP','{}+G-MF'.format(GP_RR),'{}+MF3'.format(MB_RR),
'{}+MF'.format(MB_RR),'{}+GP'.format(MB_RR),'{}+MF'.format(GP_RR),'Forgetful-MB+MF+AP']

for i in range(len(BICs)):
	print('{} had delta-BIC = {}\n'.format(df['Model Name'][i],BICs[i]))

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 6))



# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="Change in iBIC from Winning Model: \n {}+MF+GP".format(MB_RR), y="Model Name", data=df,
            label="Total", color="black")



# Add a legend and informative axis label
# ax.legend(ncol=2, loc="lower right", frameon=True)
# ax.set(xlim=(0, 24), ylabel="",
#        xlabel="Automobile collisions per billion miles")
# sns.despine(left=True, bottom=True)
plt.savefig("all_models_BICs.png",bbox_inches='tight', dpi=300)
plt.show()
