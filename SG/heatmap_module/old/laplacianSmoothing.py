#!/usr/bin/env python
# coding: utf-8
#############################
# As described here: https://liqimai.github.io/blog/AAAI-18/
############################
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
from scipy.sparse import identity
############################################
W = sparse.load_npz(sys.argv[1]) # adj.npz
npyfilename = sys.argv[2] # 'localdata.npy'
radius = int(sys.argv[3]) # the max radious for smoothing

localdata = np.load(npyfilename,allow_pickle=True) 
localdata = normalize(localdata, norm='l1', axis=0) #create the col-stochastic matrix

L = nx.laplacian_matrix(nx.from_scipy_sparse_matrix(W,edge_attribute='weight'))
del W

gamma = 1.0
smoothing = identity(L.shape[0]) - gamma*L
del L

smooth = np.zeros((localdata.shape[0],localdata.shape[1],3))
for counter in range(radius):
    localdata = smoothing.dot(localdata)
    localdata = normalize(localdata, norm='l1', axis=0) #create the col-stochastic matrix
    if counter == round((radius)/100)-1:
        smooth[:,:,0] = localdata
    if counter == round((radius)/10)-1:
        smooth[:,:,1] = localdata
    if counter == radius-1:
        smooth[:,:,2] = localdata

np.save(npyfilename+'.smooth',smooth)

