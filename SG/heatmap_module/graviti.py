#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse, linalg
import networkx as nx
from sklearn import preprocessing
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.preprocessing import normalize
import numba

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

def covd(features,G,threshold,quantiles,node_color):
    L = nx.laplacian_matrix(G) 
    delta_features = L.dot(features)
    data = np.hstack((features,delta_features)) #it has 16 features

    covdata = [] # will contain a list for each quantile
    graph2covd = []
    for q in range(quantiles):
        covq = [] # will contain a covmat for each connected subgraph
        nodes = [n for n in np.where(node_color == q)[0]]
        subG = G.subgraph(nodes)
        graphs = [g for g in list(nx.connected_component_subgraphs(subG)) if g.number_of_nodes()>=threshold] # threshold graphs based on their size
        print('The number of connected components is',str(nx.number_connected_components(subG)), ' with ',str(len(graphs)),' large enough')
        g_id = 0
        for g in graphs:
            nodeset = list(g.nodes)
            dataset = data[nodeset]
            covmat = np.cov(dataset,rowvar=False)
            covq.append(covmat)

            quant_graph = list([(q,g_id)])
            tuple_nodes = [tuple(g.nodes)]
            new_graph2covd = list(zip(quant_graph,tuple_nodes))
            graph2covd.append(new_graph2covd)
            g_id += 1
            
        covdata.append(covq)
    return covdata, graph2covd

def logdet_div(X,Y): #logdet divergence
    (sign_1, logdet_1) = np.linalg.slogdet(0.5*(X+Y)) 
    (sign_2, logdet_2) = np.linalg.slogdet(np.dot(X,Y))
    return np.sqrt( sign_1*logdet_1-0.5*sign_2*logdet_2 )

def airm(X,Y): #affine invariant riemannian metric
    A = np.linalg.inv(linalg.sqrtm(X))
    B = np.dot(A,np.dot(Y,A))
    return np.linalg.norm(linalg.logm(B))
