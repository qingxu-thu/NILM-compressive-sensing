import numpy as np
import copy
import math
 
def noisyCount(sensitivety,epsilon):
    beta = sensitivety/epsilon
    '''
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta*np.log(1.-u2)
    else:
        n_value = beta*np.log(u2)
    #print(n_value)
    '''
    n_value = np.random.laplace(0,beta)
    #print("laplace",n_value)
    return n_value
 
def laplace_mech(data,sensitivety,epsilon):
    #print(sensitivety,epsilon)
    for i in range(len(data)):
        k = data[i]
        data[i] += noisyCount(sensitivety,epsilon)
        #print(data[i])
        #print(i)
        while data[i]<0:
            data[i] = k+noisyCount(sensitivety,epsilon)
    return data

def gauss_noisyCount(sensitivety,epsilon,delta):
    #beta = sensitivety/epsilon
    #delta = 0.5
    beta = math.sqrt(2*(sensitivety*sensitivety))/epsilon
    '''
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta*np.log(1.-u2)
    else:
        n_value = beta*np.log(u2)
    #print(n_value)
    '''
    n_value = np.random.normal(0,beta)
    #print(n_value)
    return n_value


def staircase_noisyCount(sensitivety,epsilon):
    gamma = math.exp(-epsilon)/2
    S = np.random.choice([1,-1],p=[0.5,0.5])
    #print(math.exp(-epsilon))
    G = np.random.geometric(1-math.exp(-epsilon))
    #print("G",G)
    U = np.random.uniform(0,1,1)
    B = np.random.choice([0,1],p=[gamma/(gamma+(1-gamma)*math.exp(-epsilon)),1-gamma/(gamma+(1-gamma)*math.exp(-epsilon))])
    n_value = S*((1-B)*((G+gamma*U)*sensitivety)+B*((G+gamma+(1-gamma)*U)*sensitivety))
    #print("staircase",n_value)
    return n_value

def staircase_mech(data,sensitivety,epsilon):
    for i in range(len(data)):
        k = data[i]
        data[i] += staircase_noisyCount(sensitivety,epsilon)
        while data[i]<0:
            data[i] = k+staircase_noisyCount(sensitivety,epsilon)
    return data



def gauss_mech(data,sensitivety,epsilon,delta):
    #print(sensitivety,epsilon)
    for i in range(len(data)):
        k = data[i]
        #print(gauss_noisyCount(sensitivety,epsilon,delta))
        data[i] += gauss_noisyCount(sensitivety,epsilon,delta)
        while data[i]<0:
            data[i] = k+gauss_noisyCount(sensitivety,epsilon,delta)
    return data



def laplace_generator(y,sensitivety,epsilon):
    data = laplace_mech(y,sensitivety,epsilon)
    return data


def guassian_generator(y,sensitivety,epsilon,delta):
    #data = np.random.normal(0,sensitivety,epsilon)
    data = gauss_mech(y,sensitivety,epsilon,delta)
    return data

def stair_generator(y,sensitivety,epsilon):
    #print(data)
    data = staircase_mech(y,sensitivety,epsilon)
    return data


def battery_generator(y,sensitivety,epsilon,capcaity,cost_fun):
    y_0 = copy.deepcopy(y)
    data = laplace_mech(y_0,sensitivety,epsilon)
    temp = 0
    cost = 0
    for i in range(len(data)):
        data[i] = data[i] - y[i]
        temp+=data[i]
        if temp>=capcaity:
            data[i]=data[i]-temp-capcaity
            temp = capcaity
        if temp<=0:
            data[i]=data[i]-temp
            temp=0
        if i>=1:
            cost += cost_fun*(data[i]-data[i-1])
    return y_0, cost
