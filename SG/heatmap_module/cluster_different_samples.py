#!/usr/bin/env python

#################################################################
# Cluster different samples together
#################################################################

import networkx as nx
import numpy as np
from graviti import *
import sys 
from scipy import sparse, linalg
import os
import copy
import seaborn as sns; sns.set()

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

dirname = sys.argv[1]           # the directory containing the logvec files

####################################################################################################
# load logvec from different samples
###################################################################################################
import glob

ids = [ logvec_name.split('.')[0] for logvec_name in glob.glob(dirname+r'/*.logvec.npy') ]
label_set = [ int(logvec_name.split('/')[-1]) for logvec_name in set(ids) ]

logvec_list = []
dim_list = [] # contains the size of each sample in the global logvec
print('The label set is:')
print(label_set)
for id_name in set(ids):
    dim_id = 0
    for logvec_name in glob.glob(id_name+r'*.logvec.npy'):
        logvec = np.load(logvec_name)
        dim_id += logvec.shape[0]
        logvec_list.append(logvec)
    dim_list.append(dim_id)     

label_idx = 0
labels = np.zeros((0,1))
for dim in dim_list:
    array = label_set[label_idx]*np.ones((dim,1))
    labels = np.vstack((labels,array))
    label_idx += 1
####################################################################################################
# Cluster the covariance descriptors
###################################################################################################

print('Clustering the descriptors')
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.cluster import OPTICS

print('...create the clusterable embedding...')
outfile_clusterable_embedding = os.path.join(dirname, 'clusterable_embedding.merge-samples.npy')

X = np.vstack(logvec_list) # create the array of vectorized covd data
clusterable_embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(X) # this is used to identify clusters

np.save( outfile_clusterable_embedding, np.hstack((clusterable_embedding,labels)) )

print('The embedding has shape '+str(clusterable_embedding.shape))
