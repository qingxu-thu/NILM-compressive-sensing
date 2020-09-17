import time
import pandas as pd
import numpy as np
from six import iteritems
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
import nilmtk.utils
import copy
from laplace_generator import laplace_generator
from covert_redd import convert_redd
#convert_redd('C:/Users/20552/Desktop/HMM/HMM/NMLTK/low_freq/','C:/Users/20552/Desktop/HMM/HMM/NMLTK/redd.h5')

def laplace(test_elec,ep):
    num = test_elec.shape[0]
    array = np.zeros(num)
    for i in range(num):
        array[i] = test_elec.iloc[i]
    print(array)
    n = 10
    test_fix = laplace_generator(array,n,ep)
    for i in range(num):
       test_elec.iloc[i] = test_fix[i]
    return test_elec


def predict(clf, test_elec, sample_period, timezone,ep):
    pred = {}
    gt= {}
    for i, chunk in enumerate(test_elec.mains().load(physical_quantity='power',ac_type='apparent',sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()
        chunk_drop_na = laplace(chunk_drop_na,ep)
        print(chunk_drop_na)
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        gt[i]={}
        for meter in test_elec.submeters().meters:
            # Only use the meters that we trained on (this saves time!)
            gt[i][meter] = next(meter.load(physical_quantity='power',ac_type='active',sample_period=sample_period))
        gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i]) if len(v)}, index=next(iter(gt[i].values())).index).dropna()
    # If everything can fit in memory
    gt_overall = pd.concat(gt)
    gt_overall.index = gt_overall.index.droplevel()
    pred_overall = pd.concat(pred)
    pred_overall.index = pred_overall.index.droplevel()
    #print(gt_overall.index,pred_overall.index)
    # Having the same order of columns
    gt_overall = gt_overall[pred_overall.columns]
    #Intersection of index
    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)
    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.loc[common_index_local]
    pred_overall = pred_overall.loc[common_index_local]
    appliance_labels = [m for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    #print(gt_overall, pred_overall)
    return gt_overall, pred_overall

def performance(rmse,ep,temp,num):
    f = open("D:/project/NMLTK/test_real.txt",'a')
    f.write(str(ep)+"\n")
    temp += (1-rmse)
    input = "point" + str(num)+"\n"+"acc"+str(1-rmse)+"accummlate"+"\n"+str(temp)+"\n"+"average"+str(temp/(num+1))+"\n"
    f.write(input)
    '''
    if not temp:
        for i in range(len(rmse)):
            temp.append(rmse[i])
            input = "point" + str(num)+"\n"+"rmse"+str(rmse[i])+"accummlate"+"\n"+str(temp[-1])+"\n"+"average"+str(temp[i]/(num+1))+"\n"
            f.write(input)
    else:
        for i in range(len(rmse)):
            temp[i] = temp[i]+rmse[i]
            input = "point" + str(num)+"\n"+"rmse"+str(rmse[i])+"accummulate"+"\n"+str(temp[i])+"\n"+"average"+str(temp[i]/(num+1))+"\n"
            f.write(input)
    '''
    f.close()
    return temp


def perform_metric(gt, predictions):
    error = 0
    gtp = gt.values
    pred = predictions.values
    for i in range(gtp.shape[0]):
        for j in range(gtp.shape[1]):
            #print(gtp[i][j])
            if abs(gtp[i][j]-pred[i][j]) > 10:
                error += 1
    return error/(gtp.shape[0]*gtp.shape[1])

import numpy.random
numpy.random.seed(42)

n=10
print("....")
train = DataSet('D:/project/NMLTK/redd.h5')
test = DataSet('D:/project/NMLTK/redd.h5')
test_use = DataSet('D:/project/NMLTK/redd.h5')
building = 1
train.set_window(end="2011-04-30")
test.set_window(start="2011-04-30")
test_use.set_window(start="2011-04-30")
train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec
#test_use_elec = test_use.buildings[1].elec
print(test_elec)
#top_5_train_elec = train_elec.submeters().select_top_k(k=10)
top_5_train_elec = train_elec.submeters().select_top_k(k=5)

classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}
#classifiers = {'FHMM':FHMM(),'CO':CombinatorialOptimisation()}
predictions = {}
sample_period = 5
temp = 0
for clf_name, clf in classifiers.items():
    print("*"*20)
    print(clf_name)
    print("*" *20)
    clf.train(top_5_train_elec, sample_period=sample_period)
    for i in range(10):
        ep=1-0.1*i
        #ep = 10000
        #test_elec = laplace(test_elec,ep)
        #temp=[]
        for j in range(10):
            gt, predictions[clf_name] = predict(clf, test_elec,5, train.metadata['timezone'],ep)
            error = perform_metric(gt, predictions[clf_name])
            #rmse = {}
            #rmse = nilmtk.utils.compute_rmse(gt, predictions[clf_name], pretty=True)
            print(1-error)
            temp = performance(error,ep,temp,j)


