# Fuctions of different files 

1. compressive_sensing/compress_sensing.py <br>
Work for one-shot compressive sensing with different sparsity <br>
2. compressive_sensing/compress_sensing_guassian.py <br>
Work for multi-shot compressive sensing with different noise <br>
3. compressive_sensing/compress_sensing_mutishot.py <br>
Work for multi-shot compressive sensing <br>
4. compressive_sensing/laplace_generator.py <br>
Noise generator <br>
4. compressive_sensing/parse.py,compressive_sensing/parseways.py,compressive_sensing/plot.py <br>
Results analysis <br>
5.other methods/NILM_TCAS <br>
AILP and IP for NILM <br>
6.other methods/NILMTK <br>
CO and FHMM for NILM <br>
7.other methods/RNN <br>
RNN for NILM <br>
8.other methods/SparseNILM-master <br>
SparseHMM for NILM


# Environment requirement

NILMTK; keras; matlab; cvxpy


# test requrirement

1.compressive sensing:<br>
data.xlsx: meter data<br>
p_matrix.xlsx: power data<br>
ground_truth.xlsx: ground truth varible matrix<br>
2.ALIP IP:<br> 
REDDhouse1_3sec_VA: REDD house appliances data<br>
3.RNN:<br>
redd.h5: redd dataset h5 format<br>
4.SparseHMM:<br>
REDD house appliances data<br>

# some simple command

1.compressive sensing:<br>
`python compress_sensing.py`<br>
2.RNN:<br>
`python rnndisaggregator.py`<br>
`python redd-test.py`<br>
3.SparseHMM:<br>
`bash batch_BuildModels`<br>
`bash batch_TestSparse`<br>
4.ALIP or IP:<br>
`matlab TCAS2Exp2_REDD1.m`

# Reference codes
1. 
