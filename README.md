# Fuctions of different files 

1. compressive_sensing/compress_sensing.py
Work for one-shot compressive sensing with different sparsity
2. compressive_sensing/compress_sensing_guassian.py
Work for multi-shot compressive sensing with different noise
3. compressive_sensing/compress_sensing_mutishot.py
Work for multi-shot compressive sensing
4. compressive_sensing/laplace_generator.py
Noise generator
5. compressive_sensing/parse.py compressive_sensing/parseways.py compressive_sensing/plot.py
Results analysis
6. other methods/NILM_TCAS
AILP and IP for NILM
7. other methods/NILMTK
CO and FHMM for NILM
8. other methods/RNN 
RNN for NILM
9. other methods/SparseNILM-master
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
```properties
python compress_sensing.py
```
2.RNN:<br>
```properties
python rnndisaggregator.py
python redd-test.py
```
3.SparseHMM:<br>
```properties
bash batch_BuildModels
bash batch_TestSparse
```
4.ALIP or IP:<br>
```properties
matlab TCAS2Exp2_REDD1.m
```

# Reference codes

1.https://github.com/smakonin/SparseNILM
Makonin S, Popowich F, Bajić I V, et al. Exploiting HMM sparsity to perform online real-time nonintrusive load monitoring[J]. IEEE Transactions on smart grid, 2015, 7(6): 2575-2585.

2.https://github.com/OdysseasKr/neural-disaggregator
Kelly J, Knottenbelt W. Neural nilm: Deep neural networks applied to energy disaggregation[C]//Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments. 2015: 55-64.

3.http://www.sfu.ca/~ibajic/software/NILM-TCAS.rar
Bhotto M Z A, Makonin S, Bajić I V. Load disaggregation based on aided linear integer programming[J]. IEEE Transactions on Circuits and Systems II: Express Briefs, 2016, 64(7): 792-796.

# Contact us

If there are some problems in the codes' running, you're welcome to show it in issues or contact us with wanghaox19@mails.tsinghua.edu.cn.
