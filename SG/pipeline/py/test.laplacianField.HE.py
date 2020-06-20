#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys  
sys.path.insert(0, '../py')
from graviti import *

from numpy.linalg import norm
import numpy as np
import os
import os.path
from os import path
import sys
import glob
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import hdbscan
import pandas as pd
import umap
import networkx as nx
from scipy import sparse, linalg
import pickle
from sklearn.preprocessing import normalize, scale
from scipy.sparse import find
from numpy.linalg import norm
import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# In[8]:


size = 0 # number of nuclei, use 0 value for full set
nn = 20 # set the number of nearest neighbor in the umap-graph. Will be used in CovD as well


# In[9]:


for file in glob.glob('/media/garner1/hdd2/tcga.detection/*.svs.Detections.txt.gz'):
# for file in [sys.argv[1]]:
    
    sample = os.path.basename(file).split(sep='.')[0]; print(sample)

    df = pd.read_csv(file,sep='\t')

    features = df.columns[7:]
    centroids = df.columns[5:7]

    # filter by percentiles in morphologies (hardcoded in function filtering) and introduce coeff. of var
    if size == 0:
        fdf = df # filter out extremes in morphology
    else:
        fdf = df.sample(n=size) # filter out morphological outlyers and subsample nuclei
    pos = fdf[centroids].to_numpy() # Get the positions of centroids 
    fdf = fdf.rename(columns={"Centroid X µm": "cx", "Centroid Y µm": "cy"})
    # Building the UMAP graph
    filename = '../npz/'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.graph.HE.npz' # the adj sparse matrix
    if path.exists(filename):
        print('The graph already exists')
        A = sparse.load_npz(filename) 
    else:
        print('Creating the graph')
        A = space2graph(pos,nn)
        sparse.save_npz(filename, A)


    # filename = '../pkl/'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.graph.HE.pickle'    # the networkx obj
    # if path.exists(filename):    
    #     print('The network already exists')
    #     G = nx.read_gpickle(filename)
    # else:
    #     print('Creating the network')
    #     G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    #     nx.write_gpickle(G, filename)

    print('Loading data')
    data = scale(fdf[features].to_numpy(), with_mean=False) #get the morphological data and rescale the data by std 
    print('Data loaded')    
    # Get info about the graph
    row_idx, col_idx, values = find(A) 

    # Get numb of cores
    num_cores = multiprocessing.cpu_count() # numb of cores

    # Parallel generation of the local covd
    filename = '../pkl/'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.descriptor.HE.pickle'    # the descriptor
    if path.exists(filename):    
        print('The descriptor already exists')
        descriptor = pickle.load( open( filename, "rb" ) )
    else:
        print('Generating the descriptor')
        processed_list = Parallel(n_jobs=num_cores)(delayed(covd_parallel)(r,data,row_idx,col_idx) 
                                                                for r in tqdm(range(A.shape[0]))
                                                       )

        # Construct the descriptor array
        descriptor = np.zeros((len(processed_list),processed_list[0][1].shape[0]))
        for r in range(len(processed_list)):
            descriptor[r,:] = processed_list[r][1] # covd descriptors of the connected nodes
        pickle.dump( descriptor, open( filename, "wb" ), protocol=4 )

    
    print('Generating the diversity index')
    fdf['diversity'] = Parallel(n_jobs=num_cores)(delayed(covd_gradient_parallel)(node,
                                                                          descriptor,
                                                                          row_idx,col_idx,values) 
                               for node in tqdm(range(data.shape[0])))
    filename = '../pkl/'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.with_DiversityIndex.HE.pickle'
    fdf.to_pickle(filename, protocol=4)
    #Show contour plot
    N = 100
    filename = './'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.bin'+str(N)+'.contour.HE.sum.png'
    contourPlot(fdf,N,np.sum,filename)


