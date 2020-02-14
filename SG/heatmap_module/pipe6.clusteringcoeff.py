#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
import networkx as nx
warnings.filterwarnings('ignore')
############################################
A = sparse.load_npz(sys.argv[1])

degree = A.sum(axis=1) #calculate degree vector
AAA = A.dot(A.dot(A))  #calculate matrix power
num = np.cbrt(AAA.diagonal()).reshape(degree.shape) #cubic root of the diagonal as in the definition of clustering coeff
# denom = np.multiply(degree,degree-1) #normalize by degrees
# cc = np.divide(num,denom) #clustering coefficient
cc = num #new definition

np.save(sys.argv[1]+'.myclustering', cc)
