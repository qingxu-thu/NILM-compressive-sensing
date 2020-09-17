import time
import pandas as pd
import numpy as np
from six import iteritems
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
import nilmtk.utils
import copy
from laplace_generator import laplace_generator


iawe = DataSet('C:/Users/20552/Desktop/HMM/HMM/NMLTK/iawe.h5')
iawe.set_window(start="2013-06-20",end="2013-06-21")
elec = iawe.buildings[1].elec
print(elec.mains().power_series_all_data())
fridge = elec["fridge"]
print(next(fridge.power_series()))
for chunk_when_on in fridge.when_on(on_power_threshold = 40, ac_type = 'active'):
    print(chunk_when_on)
#print(next(elec.load()))
#for item in elec:
#    df = next(item.load(physical_quantity='power', ac_type='active'))
#    print(df)


