import numpy as np
import sys
import math
import xlrd
import time
from laplace_generator import laplace_generator
import copy
import random
import cvxpy as cvx
import time as ti
import h5py


def one_shot_compressed_sensing_decode(x0,y,p_matrix,delta):
    _app = p_matrix.shape[0]
    X = cvx.Variable((_app))
    #print(k)
    objective = cvx.Minimize(cvx.norm(X,1))
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
        return np.zeros(x0.shape[0])
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
    return X.value

def data_preprocess(x,y):
    _size = y.shape[0]-1
    _app = x.shape[1]
    delta_x = np.zeros((_size,_app))
    delta_y = np.zeros((_size))
    for i in range(_size-1):
        delta_x[i] = np.abs(x[i+1]-x[i])
        delta_y[i] = np.abs(y[i+1]-y[i])
    #print(delta_x,delta_y)
    return delta_x,delta_y,_size,_app

def perform_matrix(X_predict,ground_truth):
    _size = X_predict.shape[0]
    _app = X_predict.shape[1]
    error_total = np.sum(np.abs(X_predict-ground_truth))
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
    lenth = 400000
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
    #p = np.sort(p)
    print(p)
    return p

def read_ground_truth(name,time,app):
    data = xlrd.open_workbook(name)
    table = data.sheets()[0]
    start = 0
    time = 400000
    p = np.zeros((time,app))
    for i in range(time):
        p[i,:] = np.array(table.row_values(i*1+0), dtype = "float64")
    return p

def prob_cal(delta,x0):
    X = np.zeros((len(delta)+1,x0.shape[0]))
    X[0]=x0
    for i in range(len(delta)):
        for j in range(delta[i].shape[0]):
            if i==0:
                X[i+1][j] = x0[j]*(1-delta[i][j])+(1-x0[j])*delta[i][j]
            else:
                X[i+1][j] = X[i][j]*(1-delta[i][j])+(1-X[i][j])*delta[i][j]
    return X

def round(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j]<0 or x[i][j]>1:
                #print("x[i][j]",x[i][j])
                if x[i][j]>-0.0001 and x[i][j]<0:
                    x[i][j] = 0
                if 1-x[i][j]>-0.0001 and 1-x[i][j]<0:
                    x[i][j] = 1
            p = np.array([1-x[i][j],x[i][j]])
            x[i][j] = np.random.choice([0, 1], p = p.ravel())
    return x

def clip_and_adjust(X,y,p):
    _time = X.shape[0]
    k = np.argsort(p)
    X = round(X)
    for i in range(_time):
        exp = np.dot(X[i].T,p)
        if y[i]>exp:
            m = 0
            while(m < p.shape[0] and y[i]>exp):
                X[i][k[-m]] = 1
                exp = np.dot(X[i].T,p)
                m += 1
        elif y[i]<exp:
            m = 0
            while(m < p.shape[0] and y[i]<exp):
                X[i][k[m]] = 0
                exp = np.dot(X[i].T,p)
                m+=1
    return X


def muti_shot_cs(p,ground_truth,y_fix,x0):
    delta_x,delta_y,_size,_app = data_preprocess(ground_truth,y_fix)
    result = []
    for j in range(delta_y.shape[0]):
        result.append(one_shot_compressed_sensing_decode(delta_x[j],delta_y[j],p,delta))
    X_predict = prob_cal(result,x0)
    X_predict = clip_and_adjust(X_predict,y_fix,p)
    error = perform_matrix(X_predict,ground_truth)
    return error


if __name__ == "__main__":
    name_1 = "D:/project/CS5/CS3/CS/Viterbi - 副本/data.xlsx"
    name_2 = "D:/project/CS5/CS3/CS/Viterbi - 副本/p_matrix.xlsx"
    name_3 = "D:/project/CS5/CS3/CS/Viterbi - 副本/ground_truth.xlsx"
    y = read_time(name_1)
    time = y.shape[0]
    p = read_p(name_2)
    app = p.shape[0]
    ground_truth = read_ground_truth(name_3,time,app)
    y_init = copy.deepcopy(y)
    y_fix_use = copy.deepcopy(y)
    delta = 200
    n = 0.5*delta
    ep = 2.5
    a = 0
    for j in range(200):
        a = 0
        c = 0
        #ep = 100-j*0.8
        ep = 0.001*(j+1)+0.037
        k=0
        for i in range(1):
            f1 = open("D:/project/CS5/CS3/CS/Viterbi - 副本/test_realdata28.txt",'a')
            f = open("D:/project/CS5/CS3/CS/Viterbi - 副本/test_realdata29.txt",'a')
            time_start = ti.time()
            #print(y_fix_use)
            #y_fix = y_fix_use
            y_fix = laplace_generator(y_fix_use,n,ep)
            #print(y_fix)
            x0 = ground_truth[0]
            error = muti_shot_cs(p,ground_truth,y_fix,x0)
            acc = 1-error
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


