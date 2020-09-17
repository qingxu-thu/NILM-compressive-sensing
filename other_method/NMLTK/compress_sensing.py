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
import tables

def one_shot_compressed_sensing_decode(x0,y,p_matrix,delta):
    _size = y.shape[0]
    _app = p_matrix.shape[0]
    use = []
    use.append(x0)
    X = cvx.Variable((_app))
    #print(k)
    objective = cvx.Minimize(cvx.norm(X,1))
    #if abs(k)>p_matrix[-1]:
    #    constraints = [(X.T*p_matrix-k)<=delta, (X.T*p_matrix-k)>=-delta]
    #else:
    #constraints = [(X.T*p_matrix-k)<=delta, (X.T*p_matrix-k)>=-delta,cvx.max(X)<=1,cvx.min(X)>=-1]
    constraints = [(X.T*p_matrix-k)<=delta, (X.T*p_matrix-k)>=-delta,cvx.max(X)<=1,cvx.min(X)>=-1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.ECOS_BB)
    if X.value is None:
        X = cvx.Variable((_app))
        #constraints = [(X.T*p_matrix-k)<=20, (X.T*p_matrix-k)>=-20,cvx.max(X)<=1,cvx.min(X)>=-1]
        #constraints = [(X.T*p_matrix-k)<=20, (X.T*p_matrix-k)>=-20]
        constraints = [cvx.max(X)<=1,cvx.min(X)>=-1]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.ECOS_BB)
        if X.value is None:
            return []
    print(X.value)
    return np.abs(x0-X.value)

def data_preprocess(x,y):
    _size = y.shape[0]-1
    _app = x.shape[1]
    delta_x = np.zeros((_size,_app))
    delta_y = np.zeros((_size))
    for i in range(_size-1):
        delta_x[i] = x[i+1]-x[i]
        delta_y[i] = y[i+1]-y[i]
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

def load_data(path):
    with h5py.File(path,"r") as f:
        for key in f["building1"]["elec"].keys():
            for item in f["building1"]["elec"][str(key)].keys():
                if item=="table":
                    print(f["building1"]["elec"][str(key)]["table"])

load_data("C:/Users/20552/Desktop/HMM/HMM/NMLTK/iawe.h5")
