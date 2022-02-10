#function to sample parameters within a model
from numpy.random import beta,gamma,chisquare,normal,poisson,uniform,logistic,multinomial,binomial
import numpy as np
def sample_parameters(distribution_type,hyperparameter_list,sample_size):
    counter=1
    for num in hyperparameter_list:
        exec("global hp_{}; hp_{}={}".format(counter,counter,num))
        counter+=1
    exec("sample={}({},{},{})".format(distribution_type,hp_1,hp_2,sample_size))
    
    return np.array(sample)


samples=sample_parameters('beta',[2 ,2],100)
print(samples)
print(type(samples))
##x=np.arange(5,10,0.01)
##y=np.array([0,1,2,3])
##print(x[y])
