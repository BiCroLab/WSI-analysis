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
import igraph
import pandas as pd

def filtering(df):
    #First removing columns
    filt_df = df[["area","perimeter","solidity","eccentricity","circularity","mean_intensity","std_intensity"]]
    df_keep = df.drop(["area","perimeter","solidity","eccentricity","circularity","mean_intensity","std_intensity"], axis=1)
    #Then, computing percentiles
    low = .01
    high = .99
    quant_df = filt_df.quantile([low, high])
    #Next filtering values based on computed percentiles
    filt_df = filt_df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                        (x < quant_df.loc[high,x.name])], axis=0)
    #Bringing the columns back
    filt_df = pd.concat( [df_keep,filt_df], axis=1 )
    filt_df['cov_intensity'] = filt_df['std_intensity']/filt_df['mean_intensity']
    #rows with NaN values can be dropped simply like this
    filt_df.dropna(inplace=True)
    return filt_df

def space2graph(positions,nn):
    XY = positions#np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6))
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
    return mat_XY

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

def covd(mat):
    ims = coo_matrix(mat)                               # make it sparse
    imd = np.pad( mat.astype(float), (1,1), 'constant') # path with zeros

    [x,y,I] = [ims.row,ims.col,ims.data]                # get position and intensity
    pos = np.asarray(list(zip(x,y)))                    # define position vector
    length = np.linalg.norm(pos,axis=1)                 # get the length of the position vectors
    
    Ix = []  # first derivative in x
    Iy = []  # first derivative in y
    Ixx = [] # second der in x
    Iyy = [] # second der in y 
    Id = []  # magnitude of the first der 
    Idd = [] # magnitude of the second der
    
    for ind in range(len(I)):
        Ix.append( 0.5*(imd[x[ind]+1,y[ind]] - imd[x[ind]-1,y[ind]]) )
        Ixx.append( imd[x[ind]+1,y[ind]] - 2*imd[x[ind],y[ind]] + imd[x[ind]-1,y[ind]] )
        Iy.append( 0.5*(imd[x[ind],y[ind]+1] - imd[x[ind],y[ind]-1]) )
        Iyy.append( imd[x[ind],y[ind]+1] - 2*imd[x[ind],y[ind]] + imd[x[ind],y[ind]-1] )
        Id.append(np.linalg.norm([Ix,Iy]))
        Idd.append(np.linalg.norm([Ixx,Iyy]))
    #descriptor = np.array( list(zip(list(x),list(y),list(I),Ix,Iy,Ixx,Iyy,Id,Idd)),dtype='int64' ).T # descriptor
    descriptor = np.array( list(zip(list(length),list(I),Ix,Iy,Ixx,Iyy,Id,Idd)),dtype='int64' ).T     # rotationally invariant descriptor 
    C = np.cov(descriptor)            # covariance of the descriptor
    iu1 = np.triu_indices(C.shape[1]) # the indices of the upper triangular part
    covd2vec = C[iu1]
    return covd2vec


def covd_old(features,G,threshold,quantiles,node_color):
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

def get_subgraphs(G,threshold,quantiles,node_quantiles):
    subgraphs = []
    node_set = []
    for f in range(node_quantiles.shape[1]): # for every feature
        for q in range(quantiles):        # for every quantile
            nodes = [n for n in np.where(node_quantiles[:,f] == q)[0]] #get the nodes
            subG = G.subgraph(nodes) # build the subgraph
            graphs = [g for g in list(nx.connected_component_subgraphs(subG)) if g.number_of_nodes()>=threshold] # threshold connected components in subG based on their size
            subgraphs.extend(graphs)

            node_subset = [list(g.nodes) for g in graphs]
            node_set.extend(node_subset)    
    unique_nodes = list(np.unique(np.asarray([node for sublist in node_set for node in sublist])))    
            
    return subgraphs, unique_nodes

def covd_multifeature(features,G,subgraphs):
    L = nx.laplacian_matrix(G)
    delta_features = L.dot(features)
    data = np.hstack((features,delta_features)) #it has 16 features

    covdata = [] # will contain a list for each quantile
    graph2covd = []

    for g in subgraphs:
        nodeset = list(g.nodes)
        dataset = data[nodeset]
        covmat = np.cov(dataset,rowvar=False)
        covdata.append(covmat)

        graph2covd.append(list(g.nodes))
            
    return covdata, graph2covd

def community_covd(features,G,subgraphs):
    L = nx.laplacian_matrix(G)
    delta_features = L.dot(features)
    data = np.hstack((features,delta_features)) #it has 16 features

    covdata = [] # will contain a list for each community
    
    for g in subgraphs:
        nodes = [int(n) for n in g]
        dataset = data[nodes]
        covmat = np.cov(dataset,rowvar=False)
        covdata.append(covmat)
        
    return covdata

def community_covd_woLaplacian(features,G,subgraphs):
    data = features

    covdata = [] # will contain a list for each community
    
    for g in subgraphs:
        nodes = [int(n) for n in g]
        dataset = data[nodes]
        covmat = np.cov(dataset,rowvar=False)
        covdata.append(covmat)
        
    return covdata

def logdet_div(X,Y): #logdet divergence
    (sign_1, logdet_1) = np.linalg.slogdet(0.5*(X+Y)) 
    (sign_2, logdet_2) = np.linalg.slogdet(np.dot(X,Y))
    return np.sqrt( sign_1*logdet_1-0.5*sign_2*logdet_2 )

def airm(X,Y): #affine invariant riemannian metric
    A = np.linalg.inv(linalg.sqrtm(X))
    B = np.dot(A,np.dot(Y,A))
    return np.linalg.norm(linalg.logm(B))

def cluster_morphology(morphology,graph2covd,labels):
    nodes_in_cluster = []
    numb_of_clusters = len(set(labels))
    cluster_mean = np.zeros((numb_of_clusters,morphology.shape[1]))
    if -1 in set(labels):
        for cluster in set(labels):
            nodes_in_cluster.extend([graph2covd[ind] for ind in range(len(graph2covd)) if labels[ind] == cluster ])
            nodes = [item for sublist in nodes_in_cluster for item in sublist]
            ind = int(cluster)+1
            cluster_mean[ind,:] = np.mean(morphology[nodes,:],axis=0)
    else:
        for cluster in set(labels):
            nodes_in_cluster.extend([graph2covd[ind] for ind in range(len(graph2covd)) if labels[ind] == cluster ])
            nodes = [item for sublist in nodes_in_cluster for item in sublist]
            cluster_mean[cluster,:] = np.mean(morphology[nodes,:],axis=0)
    return cluster_mean

def networkx2igraph(graph,nodes,edges):     # given a networkx graph creates an igraph object
    g = igraph.Graph(directed=False)
    g.add_vertices(nodes)
    g.add_edges(edges)
    edgelist = nx.to_pandas_edgelist(graph)
    for attr in edgelist.columns[2:]:
        g.es[attr] = edgelist[attr]
    return g
