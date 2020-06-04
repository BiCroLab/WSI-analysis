#!/usr/bin/env python
# coding: utf-8

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
matplotlib.use('Agg')
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import hdbscan
import pandas as pd
import umap
from graviti import *
import networkx as nx
from scipy import sparse, linalg
import pickle
import multiprocessing
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

dirname = sys.argv[1] #'../h5/id_52/' # the path to *features.npz files 
sample = sys.argv[2] #'52' #sys.argv[2]  # the sample id
size = int(sys.argv[3]) #100000 # number of nuclei, use negative value for full set
nn = int(sys.argv[4]) #10 # set the number of nearest neighbor in the umap-graph. Will be used in CovD as well

N = 25 # number of linear bins for the contour visualization
print('N: ',str(N))

######################################
counter = 0
print('Loading the masks')
for f in glob.glob(dirname+'/*.intensity-covd.npz'): # for every fov
    counter += 1
    if counter == 1:            # set up the data arrays
        data = np.load(f,allow_pickle=True)
        fov = data['fov']
        xy = data['centroids']
        covd = data['descriptors']
    else:                       # update the data arrays
        data = np.load(f,allow_pickle=True)
        fov = np.vstack((fov,data['fov']))
        xy = np.vstack((xy, data['centroids']))
        covd = np.vstack((covd, data['descriptors']))

# Create dataframes with spatial and morphological measurements
df_fov = pd.DataFrame(data=fov, columns=['fov_row','fov_col']) # field of view dataframe
df_xy = pd.DataFrame(data=xy, columns=['cx','cy'])   # centroid dataframe
df_covd = pd.DataFrame(data=covd)

# Concatenate spatial and morphological dataframes
df = pd.concat([df_fov,df_xy, df_covd],axis=1)

if size == 0:
    fdf = df
else:
    fdf = df.sample(n=size) # filter out morphological outlyers and subsample nuclei

pos = fdf[fdf.columns[2:4]].to_numpy() # Get the positions of centroids 

# Building the UMAP graph
filename = '../py/ID'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.igraph.npz' # the adj sparse matrix
if path.exists(filename):
    print('The graph already exists')
    A = sparse.load_npz(filename) 
else:
    print('Creating the graph')
    A = space2graph(pos,nn)
    sparse.save_npz(filename, A)
    
filename = '../py/ID'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.igraph.pickle'    # the networkx obj
if path.exists(filename):    
    print('The network already exists')
    G = nx.read_gpickle(filename)
else:
    print('Creating the network')
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    nx.write_gpickle(G, filename)

descriptor = fdf[fdf.columns[4:]].to_numpy() #get the morphological data
    
# Construct the local Laplacian
L = nx.laplacian_matrix(G, weight='weight') # get the Laplacian matrix
delta_descriptor = L.dot(descriptor) # get the local differianted descriptor
delta = norm(delta_descriptor,axis=1) # get the norm of the differential field

# Contour visualization
fdf['field'] = delta # define the laplacian field
fdf['x_bin'] = pd.cut(fdf['cx'], N, labels=False) # define the x bin label
fdf['y_bin'] = pd.cut(fdf['cy'], N, labels=False) # define the y bin label

# define the pivot tabel for the contour plot
table = pd.pivot_table(fdf, 
                       values='field', 
                       index=['x_bin'],
                       columns=['y_bin'],
                       aggfunc=np.sum, # take the mean of the entries in the bin
                       fill_value=None)

X=table.columns.values
Y=table.index.values
Z=table.values
Xi,Yi = np.meshgrid(X, Y)

fig, ax = plt.subplots(figsize=(10,10))
cs = ax.contourf(Yi, Xi, Z, 
                 alpha=1.0, 
                 levels=10,
                 cmap=plt.cm.viridis);
cbar = fig.colorbar(cs)
plt.savefig('ID'+str(sample)+'.size'+str(size)+'.nn'+str(nn)+'.icontour.png')





