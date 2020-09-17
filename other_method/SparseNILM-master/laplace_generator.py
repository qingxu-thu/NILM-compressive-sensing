import numpy as np
 
def noisyCount(sensitivety,epsilon):
    beta = sensitivety/epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta*np.log(1.-u2)
    else:
        n_value = beta*np.log(u2)
    #print(n_value)
    return int(n_value)
 
def laplace_mech(data,sensitivety,epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivety,epsilon)
    return data

def laplace_generator(y,sensitivety,epsilon):
    data = laplace_mech(y,sensitivety,epsilon)
    return y