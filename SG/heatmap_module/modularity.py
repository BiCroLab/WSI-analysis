#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
import networkx as nx
warnings.filterwarnings('ignore')
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd
############################################
W = sparse.load_npz(sys.argv[1]) # adj.npz
txtfilename = sys.argv[2] # 'data.txt.gz'

degree = W.sum(axis=1) #calculate degree vector
WW = W.tocoo(copy=True)
norma = W.sum()
rowdegree = np.asarray([degree[ind] for ind in WW.row]).squeeze()
coldegree = np.asarray([degree[ind] for ind in WW.col]).squeeze()
datamodel = rowdegree*coldegree*1.0/norma
nullmodel = sparse.csr_matrix((datamodel, (WW.row, WW.col)), shape=W.shape)
M = W - nullmodel
modularity = M.sum(axis=1)

np.savetxt(txtfilename+'.nn'+str(10)+'.modularity.gz', modularity)
