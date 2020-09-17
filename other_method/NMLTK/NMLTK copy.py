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
from nilmtk.metergroup import MeterGroup, iterate_through_submeters_of_two_metergroups
from nilmtk.electric import align_two_meters
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
    for i, chunk in enumerate(test_elec.mains().load(physical_quantity='power',ac_type='active',sample_period=sample_period)):
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
    return gt_overall, pred_overall

'''
def performance(rmse,ep,temp,num):
    f = open("C:/Users/20552/Desktop/HMM/HMM/NMLTK/test_real.txt",'a')
    f.write(str(ep)+"\n")
    temp = temp+(1-rmse)
    input = "point" + str(num)+"\n"+"acc"+str(1-rmse)+"accummlate"+"\n"+"average"+str(temp/(num+1))+"\n"
    f.write(input)

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
    f.close()
    return temp
    '''


def performance(f1,acc,ep,temp,temp_1,num):
    f = open("C:/Users/20552/Desktop/HMM/HMM/NMLTK/test_real.txt",'a')
    f.write(str(ep)+"\n")
    if not temp:
        for i in range(len(f1)):
            temp.append(f1[i])
            input = "point" + str(num)+"\n"+"f1"+str(f1[i])+"accummlate"+"\n"+str(temp[-1])+"\n"+"average"+str(temp[i]/(num+1))+"\n"
            f.write(input)
            temp_1.append(acc[i])
            input = "point" + str(num)+"\n"+"acc"+str(acc[i])+"accummlate"+"\n"+str(temp_1[-1])+"\n"+"average"+str(temp_1[i]/(num+1))+"\n"
            f.write(input)
    else:
        for i in range(len(f1)):
            temp[i] = temp[i]+f1[i]
            input = "point" + str(num)+"\n"+"f1"+str(f1[i])+"accummulate"+"\n"+str(temp[i])+"\n"+"average"+str(temp[i]/(num+1))+"\n"
            f.write(input)
            temp_1[i] = temp_1[i]+acc[i]
            input = "point" + str(num)+"\n"+"acc"+str(acc[i])+"accummulate"+"\n"+str(temp_1[i])+"\n"+"average"+str(temp_1[i]/(num+1))+"\n"
            f.write(input)
    f.close()
    return temp,temp_1


def hamming_loss(predicted_state, ground_truth_state):
    num_appliances = np.size(ground_truth_state.values, axis=1)
    xors = np.sum((predicted_state.values != ground_truth_state.values),axis=1) / num_appliances
    return np.mean(xors)

def f1_score(predictions, ground_truth):
    '''Compute F1 scores.
    .. math::
        F_{score}^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    f1_scores : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the F1 score for that appliance.  If there are multiple
        chunks then the value is the weighted mean of the F1 score for 
        each chunk.
    '''
    # If we import sklearn at top of file then sphinx breaks.
    from sklearn.metrics import f1_score as sklearn_f1_score

    # sklearn produces lots of DepreciationWarnings with PyTables
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    f1_scores = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        scores_for_meter = pd.DataFrame(columns=['score', 'n_samples'])
        for aligned_states_chunk in align_two_meters(pred_meter, 
                                                     ground_truth_meter,
                                                     'when_on'):
            aligned_states_chunk.dropna(inplace=True)
            aligned_states_chunk = aligned_states_chunk.astype(int)
            score = sklearn_f1_score(aligned_states_chunk.icol(0),
                                     aligned_states_chunk.icol(1))
            scores_for_meter = scores_for_meter.append(
                {'score': score, 'n_samples': len(aligned_states_chunk)},
                ignore_index=True)

        # Calculate weighted mean
        tot_samples = scores_for_meter['n_samples'].sum()
        scores_for_meter['proportion'] = (scores_for_meter['n_samples'] / 
                                          tot_samples)
        avg_score = (scores_for_meter['score'] * 
                     scores_for_meter['proportion']).sum()
        f1_scores[pred_meter.instance()] = avg_score

    return pd.Series(f1_scores)


def acc_score(predictions, ground_truth):
    '''Compute F1 scores.
    .. math::
        F_{score}^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    f1_scores : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the F1 score for that appliance.  If there are multiple
        chunks then the value is the weighted mean of the F1 score for 
        each chunk.
    '''
    # If we import sklearn at top of file then sphinx breaks.
    from sklearn.metrics import accuracy_score as sklearn_acc_score

    # sklearn produces lots of DepreciationWarnings with PyTables
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    f1_scores = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        scores_for_meter = pd.DataFrame(columns=['score', 'n_samples'])
        for aligned_states_chunk in align_two_meters(pred_meter, 
                                                     ground_truth_meter,
                                                     'when_on'):
            aligned_states_chunk.dropna(inplace=True)
            aligned_states_chunk = aligned_states_chunk.astype(int)
            score = sklearn_acc_score(aligned_states_chunk.icol(0),
                                     aligned_states_chunk.icol(1))
            scores_for_meter = scores_for_meter.append(
                {'score': score, 'n_samples': len(aligned_states_chunk)},
                ignore_index=True)

        # Calculate weighted mean
        tot_samples = scores_for_meter['n_samples'].sum()
        scores_for_meter['proportion'] = (scores_for_meter['n_samples'] / 
                                          tot_samples)
        avg_score = (scores_for_meter['score'] * 
                     scores_for_meter['proportion']).sum()
        f1_scores[pred_meter.instance()] = avg_score

    return pd.Series(f1_scores)




import numpy.random
numpy.random.seed(42)

n=10
print("....")
train = DataSet('C:/Users/20552/Desktop/HMM/HMM/NMLTK/iawe.h5')
test = DataSet('C:/Users/20552/Desktop/HMM/HMM/NMLTK/iawe.h5')
test_use = DataSet('C:/Users/20552/Desktop/HMM/HMM/NMLTK/iawe.h5')
building = 1
train.set_window(end="2013-07-13")
test.set_window(start="2013-07-13")
test_use.set_window(start="2013-07-13")
train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec
#test_use_elec = test_use.buildings[1].elec
print(test_elec)
#top_5_train_elec = train_elec.submeters().select_top_k(k=10)
top_5_train_elec = train_elec.submeters().select_top_k(k=5)

classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}
classifiers = {'FHMM':FHMM()}
predictions = {}
sample_period = 5
for clf_name, clf in classifiers.items():
    print("*"*20)
    print(clf_name)
    print("*" *20)
    clf.train(top_5_train_elec, sample_period=sample_period)
    for i in range(50):
        ep=1-i*0.02
        #test_elec = laplace(test_elec,ep)
        temp=[]
        temp_1=[]
        for j in range(5):
            gt, predictions[clf_name] = predict(clf, test_elec,5, train.metadata['timezone'],ep)
            rmse = {}
            #rmse = nilmtk.utils.compute_rmse(gt, predictions[clf_name], pretty=True)
            #distance = hamming_loss(gt,predictions[clf_name])
            my_f1_score = f1_score(gt,predictions[clf_name])
            acc = acc_score(gt,predictions[clf_name])
            print(my_f1_score,acc)
            temp,temp_1 = performance(my_f1_score,acc,ep,temp,temp_1,j)



