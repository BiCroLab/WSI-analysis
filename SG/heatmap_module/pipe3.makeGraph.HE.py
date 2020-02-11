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

    nn = int(np.log2(XY.shape[0])) #scale the nn logarithmically in the numb of nodes to have enough density of edges for clustering
    print('UMAP with nn='+str(nn))
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
    sparse.save_npz(sys.argv[1]+'_graph.npz',mat_XY)
    
if __name__=="__main__":
    main()



