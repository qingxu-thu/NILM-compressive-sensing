from matplotlib import pyplot as plt
import xlrd
import numpy as np

def read(name):
    data = xlrd.open_workbook(name)
    table = data.sheets()[2]
    x = np.array(table.col_values(0), dtype = "float64")
    y_1 = np.array(table.col_values(1), dtype = "float64")
    y_2 = np.array(table.col_values(2), dtype = "float64")
    return x,y_1,y_2

x,y_1,y_2= read("C:/Users/20552/Desktop/HMM/HMM/NMLTK/result1.xlsx")
h2, = plt.plot(x,y_2,label="CO")
#h2 = plt.fill_between(x,y_h,y_l,alpha=0.4,label="Percent 5%-95%")
plt.ylim(0.35,0.58)
plt.xlabel("ln(1/$\epsilon$)",weight="black")
plt.ylabel("Accuracy",weight="black")
h1, = plt.plot(x,y_1,label="FHMM")
#h4, = plt.plot(x,y_3,label="Upper Bound")
#hh = [h1,h2,h3,h4]
plt.legend([h1,h2],["CO","FHMM"])
#plt.legend([h1,h2,h3,h4],["Actual Performance","Percent 5%-95%","Lower Bound","Upper Bound"])
#plt.plot(x,y_h,alpha=0.01)
#plt.plot(x,y_l,alpha=0.01)
#plt.legend(["Actual Performance","Percent 5%-95%","Lower Bound","Fitting Mean Performance"])
#plt.legend(["Actual Results","Theory Results","Fitting Curve"])
#plt.legend("Precent 5\%-95\%",["Precent 5\%-95\%"])
plt.savefig("C:/Users/20552/Desktop/HMM/HMM/NMLTK/fig.pdf", format='pdf',bbox_inches = 'tight',dpi=300)
