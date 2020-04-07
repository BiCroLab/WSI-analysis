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
npyfilename = sys.argv[2] # 'localdata.npy'

localdata = np.load(npyfilename,allow_pickle=True) 

L = nx.laplacian_matrix(nx.from_scipy_sparse_matrix(W,edge_attribute='weight')) 
del W
laplace = L.dot(localdata)
print(laplace.shape)

np.save(npyfilename+'.laplacian',laplace)

