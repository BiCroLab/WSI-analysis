#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
import networkx as nx
from sklearn import preprocessing
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.preprocessing import normalize

def space2graph(filename,nn):
    XY = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6))
    mat_XY = umap.umap_.fuzzy_simplicial_set(
        XY,
        n_neighbors=nn, 
        random_state=np.random.RandomState(seed=42),
        metric='l2',
        metric_kwds={},
        knn_indices=None,
        knn_dists=None,
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=2.0,
        verbose=False
    )
    return mat_XY, XY

def getdegree(graph):
    d = np.asarray(graph.degree(weight='weight'))[:,1] # as a (N,) array
    r = d.shape[0]
    return d.reshape((r,1))

def clusteringCoeff(A):
    AA = A.dot(A)
    AAA = A.dot(AA)  
    d1 = AA.mean(axis=0) 
    m = A.mean(axis=0)
    d2 = np.power(m,2)
    num = AAA.diagonal().reshape((1,A.shape[0]))
    denom = np.asarray(d1-d2)
    cc = np.divide(num,denom*A.shape[0]) #clustering coefficient
    r, c = cc.shape
    return cc.reshape((c,r))

def rescale(data):
    newdata = preprocessing.minmax_scale(data,feature_range=(-1, 1),axis=0) # rescale data so that each feature ranges in [0,1]
    return newdata

def principalComp(data):
    pca = PCA(n_components='mle')
    pca.fit(data)
    return pca

def smoothing(W,data,radius):
    S = normalize(W, norm='l1', axis=1) #create the row-stochastic matrix

    smooth = np.zeros((data.shape[0],data.shape[1]))
    summa = data
    for counter in range(radius):
        newdata = S.dot(data)
        summa += newdata
        data = newdata
        if counter == radius-1:
            smooth = summa*1.0/(counter+1)
    return smooth



