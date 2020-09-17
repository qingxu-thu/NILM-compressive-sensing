import numpy as np
import sys
import math
import xlrd
import time
from laplace_generator import laplace_generator,guassian_generator,stair_generator
import copy
import random
import cvxpy as cvx
import time as ti
import h5py

def one_shot_compressed_sensing_decode(x0,y,p_matrix,delta):
    _app = p_matrix.shape[0]
    #print(_app)
    X = cvx.Variable((_app))
    #print(k)
    objective = cvx.Minimize(cvx.norm(X,1))
    #print(y)
    #if abs(k)>p_matrix[-1]:
    #    constraints = [(X.T*p_matrix-k)<=delta, (X.T*p_matrix-k)>=-delta]
    #else:
    #constraints = [(X.T*p_matrix-k)<=delta, (X.T*p_matrix-k)>=-delta,cvx.max(X)<=1,cvx.min(X)>=-1]
    constraints = [(X.T*p_matrix-y)<=delta, (X.T*p_matrix-y)>=-delta,cvx.max(X)<=1,cvx.min(X)>=0]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.ECOS_BB)
    if X.value is None:
        #print("error")
        #print(x0.shape[0])
        return x0.shape[0]
        '''
        X = cvx.Variable((_app))
        #constraints = [(X.T*p_matrix-k)<=20, (X.T*p_matrix-k)>=-20,cvx.max(X)<=1,cvx.min(X)>=-1]
        #constraints = [(X.T*p_matrix-k)<=20, (X.T*p_matrix-k)>=-20]
        constraints = [cvx.max(X)<=1,cvx.min(X)>=-1]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.ECOS_BB)
        if X.value is None:
            return []
        '''
    #print(X.value)
    return np.sum(np.abs(x0-X.value))

def data_preprocess(x,y):
    _size = y.shape[0]
    #print(_size)
    _app = x.shape[1]
    delta_x = np.zeros((_size,_app))
    delta_y = np.zeros((_size))
    for i in range(_size-1):
        delta_x[i] = np.abs(x[i+1]-x[i])
        delta_y[i] = np.abs(y[i+1]-y[i])
    #print(delta_x,delta_y)
    return delta_x,delta_y,_size,_app

def perform_matrix(error,_size,_app):
    error_total = 0
    for item in error:
        error_total+=item
    print(error_total/(_size*_app))
    return error_total/(_size*_app)

def theory_curve(c_p,delta,elp):
    _num = elp.shape[0]
    low_bound = np.zeros((_num))
    upper_bound = np.zeros((_num))
    for i in range(_num):
        low_bound = 1
        upper_bound = 1
    return low_bound,upper_bound

def read_time(name):
    data = xlrd.open_workbook(name)
    table = data.sheets()[0]
    start = 0
    lenth = 1000
    data_use = np.zeros((lenth)) 
    time = np.array(table.col_values(0), dtype = "float64")
    for i in range(lenth):
        #print(time[start+8*i])
        data_use[i] = time[start+i]
    return data_use

def read_p(name):
    data = xlrd.open_workbook(name)
    table = data.sheets()[0]
    p = np.array(table.col_values(0), dtype = "float64")
    p = np.sort(p)
    print(p)
    return p

def read_ground_truth(name,time,app):
    data = xlrd.open_workbook(name)
    table = data.sheets()[0]
    start = 2
    time = 1000
    p = np.zeros((time,app))
    for i in range(time):
        p[i,:] = np.array(table.row_values(i+0), dtype = "float64")
    return p

def zero_one_seq(sparsity,num,app):
    ground_truth = np.zeros((app,num))
    for j in range(app):
        interval = np.random.choice(2, num, p=[sparsity, 1-sparsity])
        gt = np.zeros((num))
        for i in range(num):
            if i==0:
                gt[i]=interval[i]
            else:
                if interval[i]==1:
                    gt[i] = abs(gt[i-1]-1)
                else:
                    gt[i] = gt[i-1]
        ground_truth[j]=gt
    return ground_truth

def synatic_data(sparsity,num):
    name_2 = "D:/project/CS5/CS8/CS/Viterbi - 副本/p_sy.xlsx"
    p = read_p(name_2)
    print(p)
    app = p.shape[0]
    p = p.reshape([p.shape[0],1])
    ground_truth = zero_one_seq(sparsity,num,app)
    y = (ground_truth.T@p)
    print(y.shape)
    y = y.reshape([y.shape[0]])
    p = p.reshape([p.shape[0]])
    return y,p, ground_truth.T


if __name__ == "__main__":

    name_1 = "D:/project/CS5/CS8/CS/Viterbi - 副本/data.xlsx"
    name_2 = "D:/project/CS5/CS8/CS/Viterbi - 副本/p_matrix.xlsx"
    name_3 = "D:/project/CS5/CS8/CS/Viterbi - 副本/ground_truth.xlsx"
    y = read_time(name_1)
    time = y.shape[0]
    p = read_p(name_2)
    app = p.shape[0]
    ground_truth = read_ground_truth(name_3,time,app)
    
    sparsity = 0.2
    num = 600
    #y,p,ground_truth = synatic_data(sparsity,600)
    time = y.shape[0]
    app = p.shape[0]  
    print(time)
    y_init = copy.deepcopy(y)
    y_fix_use = copy.deepcopy(y)
    delta = 200
    n = 0.5*delta
    ep = 2.5
    a = 0
    for j in range(100):
        a = 0
        c = 0
        #ep = 100-j*0.8
        #ep = 100
        ep = 0.01+0.001*(j+1)
        k=0
        prob_delta = 0.5
        for i in range(1):
            f1 = open("D:/project/CS5/CS8/CS/Viterbi - 副本/gussian_data02.txt",'a')
            f = open("D:/project/CS5/CS8/CS/Viterbi - 副本/gussian_data02_2.txt",'a')
            time_start = ti.time()
            #print(y_fix_use)
            #y_fix = y_fix_use
            y_fix = guassian_generator(y_fix_use,n,ep,prob_delta)
            #print(y_fix)
            delta_x,delta_y,_size,_app = data_preprocess(ground_truth,y_fix)
            error = []
            for j in range(delta_y.shape[0]):
                error.append(one_shot_compressed_sensing_decode(delta_x[j],delta_y[j],p,delta))
            acc =  1-perform_matrix(error,_size,_app)
            print("acc",acc)
            temp = k
            a+=acc
            c+=1
            k = a/c
            if temp-k<0.00001 and i>10:
                break
            print("avg_acc", a/c)
            input = "point" + str(c)+"\n"+"acc"+str(a/c)+"\n"+"ep"+str(ep)+"\n"
            f.write(input)
            f.close()
            y_fix_use = copy.deepcopy(y)
        if c==0:
            print("error")
        else:
            print("final_acc", a/c, "point",n)
            input = "point" + str(c)+"\n"+"acc"+str(a/c)+"\n"+"ep"+str(ep)+"\n"
            f1.write(input)
            f1.close()

    '''
    name_1 = "D:/project/CS5/CS8/CS/Viterbi - 副本/data.xlsx"
    name_2 = "D:/project/CS5/CS8/CS/Viterbi - 副本/p_matrix.xlsx"
    name_3 = "D:/project/CS5/CS8/CS/Viterbi - 副本/ground_truth.xlsx"
    y = read_time(name_1)
    time = y.shape[0]
    p = read_p(name_2)
    app = p.shape[0]
    ground_truth = read_ground_truth(name_3,time,app)
    
    sparsity = 0.2
    num = 600
    #y,p,ground_truth = synatic_data(sparsity,600)
    time = y.shape[0]
    app = p.shape[0]  
    print(time)
    y_init = copy.deepcopy(y)
    y_fix_use = copy.deepcopy(y)
    delta = 200
    n = 0.5*delta
    ep = 2.5
    a = 0
    for j in range(100):
        a = 0
        c = 0
        #ep = 100-j*0.8
        #ep = 100
        ep = 0.01+0.001*(j+1)
        k=0
        prob_delta = 0.001
        for i in range(1):
            f1 = open("D:/project/CS5/CS8/CS/Viterbi - 副本/laplace_data02.txt",'a')
            f = open("D:/project/CS5/CS8/CS/Viterbi - 副本/laplace_data02_2.txt",'a')
            time_start = ti.time()
            #print(y_fix_use)
            #y_fix = y_fix_use
            y_fix = laplace_generator(y_fix_use,n,ep)
            #print(y_fix)
            delta_x,delta_y,_size,_app = data_preprocess(ground_truth,y_fix)
            error = []
            for j in range(delta_y.shape[0]):
                error.append(one_shot_compressed_sensing_decode(delta_x[j],delta_y[j],p,delta))
            acc =  1-perform_matrix(error,_size,_app)
            print("acc",acc)
            temp = k
            a+=acc
            c+=1
            k = a/c
            if temp-k<0.00001 and i>10:
                break
            print("avg_acc", a/c)
            input = "point" + str(c)+"\n"+"acc"+str(a/c)+"\n"+"ep"+str(ep)+"\n"
            f.write(input)
            f.close()
            y_fix_use = copy.deepcopy(y)
        if c==0:
            print("error")
        else:
            print("final_acc", a/c, "point",n)
            input = "point" + str(c)+"\n"+"acc"+str(a/c)+"\n"+"ep"+str(ep)+"\n"
            f1.write(input)
            f1.close()

    name_1 = "D:/project/CS5/CS8/CS/Viterbi - 副本/data.xlsx"
    name_2 = "D:/project/CS5/CS8/CS/Viterbi - 副本/p_matrix.xlsx"
    name_3 = "D:/project/CS5/CS8/CS/Viterbi - 副本/ground_truth.xlsx"
    y = read_time(name_1)
    time = y.shape[0]
    p = read_p(name_2)
    app = p.shape[0]
    ground_truth = read_ground_truth(name_3,time,app)
    
    sparsity = 0.2
    num = 600
    #y,p,ground_truth = synatic_data(sparsity,600)
    time = y.shape[0]
    app = p.shape[0]  
    print(time)
    y_init = copy.deepcopy(y)
    y_fix_use = copy.deepcopy(y)
    delta = 200
    n = 0.5*delta
    ep = 2.5
    a = 0
    for j in range(100):
        a = 0
        c = 0
        #ep = 100-j*0.8
        #ep = 100
        ep = 0.01+0.001*(j+1)
        k=0
        prob_delta = 0.001
        for i in range(1):
            f1 = open("D:/project/CS5/CS8/CS/Viterbi - 副本/stair_data02.txt",'a')
            f = open("D:/project/CS5/CS8/CS/Viterbi - 副本/stair_data02_2.txt",'a')
            time_start = ti.time()
            #print(y_fix_use)
            #y_fix = y_fix_use
            y_fix = stair_generator(y_fix_use,n,ep)
            #print(y_fix)
            delta_x,delta_y,_size,_app = data_preprocess(ground_truth,y_fix)
            error = []
            for j in range(delta_y.shape[0]):
                error.append(one_shot_compressed_sensing_decode(delta_x[j],delta_y[j],p,delta))
            acc =  1-perform_matrix(error,_size,_app)
            print("acc",acc)
            temp = k
            a+=acc
            c+=1
            k = a/c
            if temp-k<0.00001 and i>10:
                break
            print("avg_acc", a/c)
            input = "point" + str(c)+"\n"+"acc"+str(a/c)+"\n"+"ep"+str(ep)+"\n"
            f.write(input)
            f.close()
            y_fix_use = copy.deepcopy(y)
        if c==0:
            print("error")
        else:
            print("final_acc", a/c, "point",n)
            input = "point" + str(c)+"\n"+"acc"+str(a/c)+"\n"+"ep"+str(ep)+"\n"
            f1.write(input)
            f1.close()


    '''