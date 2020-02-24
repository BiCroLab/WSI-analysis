#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
warnings.filterwarnings('ignore')
#####################
def main():
    XY = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(5,6))
    nn = 10 # keep nn small or it will provide counterintuitive results for the clustering coefficient
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

    A = mat_XY
    degree = A.sum(axis=1) #calculate degree vector

    AA = A.dot(A)
    AAA = A.dot(AA)  
    d1 = AA.mean(axis=0) 
    m = A.mean(axis=0)
    d2 = np.power(m,2)

    num = AAA.diagonal().reshape((1,A.shape[0]))
    denom = np.asarray(d1-d2)
    cc = np.divide(num,denom*A.shape[0]) #clustering coefficient

    WW = A.tocoo(copy=True)
    norma = A.sum()
    rowdegree = np.asarray([degree[ind] for ind in WW.row]).squeeze()
    coldegree = np.asarray([degree[ind] for ind in WW.col]).squeeze()
    datamodel = rowdegree*coldegree*1.0/norma
    nullmodel = sparse.csr_matrix((datamodel, (WW.row, WW.col)), shape=WW.shape)
    M = A - nullmodel
    modularity = M.sum(axis=1) #modularity

    np.savetxt(sys.argv[1]+'.nn'+str(nn)+'.degree.gz', degree)
    sparse.save_npz(sys.argv[1]+'.nn'+str(nn)+'.adj.npz',mat_XY)
    np.savetxt(sys.argv[1]+'.nn'+str(nn)+'.cc.gz', cc)
    np.savetxt(sys.argv[1]+'.nn'+str(nn)+'.modularity.gz', modularity)
  
    
if __name__=="__main__":
    main()



