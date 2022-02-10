import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQʀₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)
  
# display subscript
MB_RR=('MB{}{}'.format(get_sub('R'),get_sub('R')))

sns.set(style='white', palette='Greys', font='arial', font_scale=2.0, rc=None)
df=pd.DataFrame()
winning_model=32709.3833984
BICs=[38009.37073804648-winning_model, 34025.5448863697-winning_model, 33896.481327053836-winning_model, 32938.79770167372-winning_model,0]
# [38009.37073804648, 34676.46504166039, 34221.15410923616, 34025.5448863697, 33896.481327053836, 32938.79770167372]
df['Change in iBIC from Winning Model']=BICs
df['Model Name'] = ['Null','MB', MB_RR, '{}+MF'.format(MB_RR), '{}+MF+GP'.format(MB_RR)]

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 6))



# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="Change in iBIC from Winning Model", y="Model Name", data=df,
            label="Total", color="black")



# Add a legend and informative axis label
# ax.legend(ncol=2, loc="lower right", frameon=True)
# ax.set(xlim=(0, 24), ylabel="",
#        xlabel="Automobile collisions per billion miles")
# sns.despine(left=True, bottom=True)
plt.savefig("changeBICs_withAPs.png",bbox_inches='tight', dpi=300)
plt.show()
