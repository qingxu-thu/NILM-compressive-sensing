import numpy as numpy
import math
#from NILM_hmm_realfix import feasible_set, judge, beta_cal, observed_prob_cal, certain_observed_prob_cal, certain_state_transform_cal

def viterbi(X, y_t, p_matrix, prob):
    t = y_t.shape[0]
    decode_seq = []
    max_seq = t*[None]
    for i in range(t):
        feasible = feasible_set(X, y_t[i])
        beta = beta_cal(feasible, y_t[i], p_matrix)
        temp = 0
        last = 0
        if i == 0:
            temp = 0
            iter_last = 0
            next_dic = dict()
            for x in feasible:
                temp = observed_prob_cal(x[0], y_t[i], p_matrix, beta, i)
                last = x
                next_dic[x[0]] = (temp, last)
        else:
            for x in feasible:
                for (j, (prob_val, last)) in enumerate(decode_seq[-1]):
                    if prob_val*observed_prob_cal(x[0], y_t[i], p_matrix, beta, i) * certain_state_transform_cal(x[0],j,prob)>temp:
                        temp = prob_val*observed_prob_cal(x[0], y_t[i], p_matrix, beta, i) * certain_state_transform_cal(x[0],j,prob)
                        last = j
                next_dic = dict()
                next_dic[x[0]] = (temp, last)
            decode_seq.append(next_dic)
            temp = 0
            iter_last = 0
            for (j, (prob_val, last)) in enumerate(decode_seq[-1]):
                if prob_val > temp:
                    temp = prob_val
                    max_seq[-1] = j
                    iter_last = last
    for i in range(t-1):
        max_seq[t-2-i] = iter_last
        iter_last = decode_seq[t-2-i][iter_last](1)
    return max_seq