import numpy as np
import os
import re
from matplotlib import pyplot as plt
import math
import matplotlib

matplotlib.rcParams['text.usetex'] = True

name_list = ["D:/project/CS5/CS8/CS/Viterbi - 副本/synctic_data020.3.txt",
            "D:/project/CS5/CS8/CS/Viterbi - 副本/synctic_data020.5.txt",
            "D:/project/CS5/CS8/CS/Viterbi - 副本/synctic_data020.7.txt",
            "D:/project/CS5/CS8/CS/Viterbi - 副本/synctic_data020.9.txt"]
#name = "D:/project/CS5/CS8/CS/Viterbi - 副本/synctic_data020.3.txt"
sparsity_list = [0.3,0.5,0.7,0.9]
for idx, name in enumerate(name_list):
    f = open(name,'r')
    lines = f.readlines()
    f.close()
    ep_list = []
    acc_list = []

    for i in range(len(lines)):
        ep_t = re.findall(r"ep(.*)",lines[i])
        #print(ep_t)
        if len(ep_t)>0:
            ep_list.append(math.log(1/float(ep_t[0])))
        acc_t = re.findall(r"acc(.*)",lines[i])
        if len(acc_t)>0:
            print(acc_t)
            acc_list.append(float(acc_t[0]))
    print(ep_list,acc_list)
    ep_array = np.array(ep_list)
    acc_array = np.array(acc_list)
    h = plt.plot(ep_array,acc_array,label="sparsity="+str(sparsity_list[idx]))

#plt.ylim(0.2,1)
plt.xlabel("DP parameter ln(1/$\epsilon$)",weight="black")
plt.ylabel("Accuracy $\\alpha$",weight="black")
plt.legend()
plt.savefig("D:/project/CS5/CS8/CS/Viterbi - 副本/sy_fig2.pdf", format='pdf',bbox_inches = 'tight',dpi=300)