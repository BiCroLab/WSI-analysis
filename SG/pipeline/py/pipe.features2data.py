#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
import glob
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import hdbscan
import pandas as pd
import umap
from graviti import *
import networkx as nx
from scipy import sparse, linalg
import warnings
warnings.filterwarnings('ignore')


dirname = sys.argv[1] # the directory where features.npz files are located
sample = sys.argv[2]  # the sample id

counter = 0
for f in glob.glob(dirname+'/*features.npz'): # for every fov
    counter += 1
    if counter == 1:            # to set up the data arrays
        data = np.load(f,allow_pickle=True)
        fov = data['fov']
#        covds = data['descriptors']
        xy = data['centroids']
        morphology = data['morphology']
    else:                       # to update the data arrays
        data = np.load(f,allow_pickle=True)
        fov = np.vstack((fov,data['fov']))
#        covds = np.vstack((covds, data['descriptors']))
        xy = np.vstack((xy, data['centroids']))
        morphology = np.vstack((morphology, data['morphology']))

# Create dataframes
df_fov = pd.DataFrame(data=fov, columns=['fov_row','fov_col'])
df_xy = pd.DataFrame(data=xy, columns=['cx','cy'])
df_morphology = pd.DataFrame(data=morphology, columns=['area','perimeter','solidity','eccentricity','circularity','mean_intensity','std_intensity'])
#df_covds = pd.DataFrame(data=covds)

# Concatenate all dataframes
df = pd.concat([df_fov,df_xy, df_morphology],axis=1)

# filter by percentiles in morphologies (hardcoded in function filtering)
fdf = filtering(df)#.sample(n=100000)

# Get the positions of centroids 
pos = fdf[fdf.columns[2:4]].to_numpy()

print('Build the UMAP graph')
A = space2graph(pos,10)
sparse.save_npz(sample+'.graph.npz', A)
G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
nx.write_gpickle(G, sample+".graph.pickle")

print('Smooth the morphology')
radius = 10000
data = fdf[fdf.columns[4:]].to_numpy()
smooth_data = smoothing(A,data,radius)
new_fdf = pd.DataFrame(data=smooth_data,columns=fdf.columns[4:],index=fdf.index)
df = pd.concat([fdf[fdf.columns[:4]],new_fdf],axis=1)

# Save dataframe
df.to_pickle("id_"+str(sample)+".measurements.smoothed.r"+str(radius)+".pkl")

# # UMAP representation of the intensity descriptors, always check the proper column selection!
# embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(df[df.columns[4:]]) 

# # # Create dataframes of the umap embedding
# df_embedding = pd.DataFrame(data=embedding, columns=['xi','yi','zi'])

# # Concatenate the embedded dataframes
# df.reset_index(drop=True, inplace=True)
# df_embedding.reset_index(drop=True, inplace=True)
# df_final = pd.concat([df, df_embedding],axis=1)

# # Save dataframe
# df_final.to_pickle("id_"+str(sample)+".measurements..pkl")

