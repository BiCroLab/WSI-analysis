#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

############################################
filename = sys.argv[1] # localdata.npy or walkhistory.npy
data = np.load(filename) 
print(data.shape)

if len(data.shape) == 3:
    for time in range(3):
        X = data[:,:,time]
        X = normalize(X, norm='l1', axis=0) #create the col-stochastic matrix
        pca = PCA(n_components='mle')
        pca.fit(X)
        # print(pca.explained_variance_ratio_)
        # print(pca.n_components_,pca.n_features_,pca.n_samples_)
        np.save(filename+'.principalComp.window-'+str(time),pca.components_)
        np.save(filename+'.singularvalues.window-'+str(time),pca.singular_values_)

if len(data.shape) == 2:
    X = data
    pca = PCA(n_components='mle')
    pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # print(pca.n_components_,pca.n_features_,pca.n_samples_)
    np.save(filename+'.principalComp',pca.components_)
    np.save(filename+'.singularvalues',pca.singular_values_)
