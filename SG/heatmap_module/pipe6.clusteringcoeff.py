#!/usr/bin/env python
# coding: utf-8

#################################
# Use the definition of clustering coefficient found in 
# https://content.iospress.com/articles/ai-communications/aic408
# (W^3)_kk / [(e^T.w_k)^2 - ||w_k||_2^2]
# where w_k is the k-th row of W
################################

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
import networkx as nx
warnings.filterwarnings('ignore')

############################################
A = sparse.load_npz(sys.argv[1]) #load the graph matrix

degree = A.sum(axis=1) #calculate degree vector
AAA = A.dot(A.dot(A))  #calculate matrix power
num = AAA.diagonal().reshape((A.shape[0],1))   #cubic root of the diagonal as in the definition of clustering coeff

d1 = np.power(A.sum(axis=1),2) # (e^T.w_k)^2
d2 = np.square(A).sum(axis=1)  # ||w_k||_2^2

cc = np.divide(num,d1-d2) #clustering coefficient

np.save(sys.argv[1]+'.cc', cc)
