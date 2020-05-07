#!/usr/bin/env python
# coding: utf-8

'''
Load the necessary libraries
'''
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px

import seaborn as sns

import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
from sklearn.metrics import pairwise_distances_argmin_min
import hdbscan
from scipy.cluster.hierarchy import fcluster

import plotly
#plotly.io.orca.config.executable = '/usr/local/share/anaconda3/bin/orca'

filename = sys.argv[1]          # pkl file 

def cluster_nuclei_intensity(filename,df,n_neighbors,threshold_q,auto_open,plot_switch):
    embedding = df[['xi','yi','zi']].to_numpy()   
    '''
    Calculate the local curvature of the point cloud embedding
    '''
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    eigvals = [LA.eigvalsh(np.cov(embedding[indices[idx,:],:].T)) for idx in range(embedding.shape[0])] #full data
    curvatures = [min(eigvals[idx])/sum(eigvals[idx]) for idx in range(len(eigvals))]
    # Add curvature to the dataframe
    df['curvature'] = curvatures 
    # Find the minima in curvature histrogram
    q1 = np.quantile(curvatures,threshold_q)
    df1 = df[df['curvature'] <= q1] # define the low curvature sector
    min_cluster_size = round(df1.shape[0]/15) # parameter to be adjausted
    min_samples = round(min_cluster_size/15)       # parameter to be adjausted

    clusterer = hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size,gen_min_span_tree=True)
    clusterer.fit(df1.loc[:,('xi','yi','zi')]) 
    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=sns.color_palette("Set2",len(clusterer.labels_)))
    plt.savefig(filename+'.tree.intensity.png')
    plt.close()
    
    df1['cluster_intensity'] = clusterer.labels_    # add cluster id to dataframe
    df1_filtered = df1[df1.cluster_intensity > -1] # remove unassigned points

    # expand the clusters to the entire point-cloud
    idx, dist = pairwise_distances_argmin_min(df[['xi','yi','zi']].to_numpy(),df1_filtered[['xi','yi','zi']].to_numpy())
    df['cluster_intensity'] = [int(df1_filtered.cluster_intensity.iloc[idx[row]])+1 for row in range(df.shape[0])] #add 1 to avoid confusion with background
    df['cluster_intensity'] = df['cluster_intensity'].apply(str)
    
    df.to_csv(filename+'.intensity.csv',index=False) # writhe to file
    if plot_switch:
        fig = px.scatter(df,#1_filtered,                                                                                                                                            
                         x="cx", y="cy",color="cluster_intensity",
                        width=800, height=800,
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker=dict(size=2,opacity=1.0))
        fig.update_layout(template='simple_white')
        fig.update_layout(legend= {'itemsizing': 'constant'})
        fig.write_image(filename+'.spatial_projection.intensity.png')
    return df
##############################################################
sample_size = 0 # set to 0 if the entire sample is considered
n_neighbors = 100   # NNN in the curvature calculation
threshold_q = 0.1   # the quantile defining the low-curvature sector
auto_open = False    # switch to open or not html figures in new tab
plot_switch = True  # switch to generate or not html figures

df = pd.read_pickle(filename)
if sample_size > 0 and sample_size < df.shape[0]:
    df = df.sample(n=sample_size)
    
cluster_nuclei_intensity(filename,df,n_neighbors=100,threshold_q=0.1,auto_open=auto_open,plot_switch=plot_switch)


# Clustering based on morphology does not provide good looking spatial projections. The umap manifold seems to be too complicated, while the intensity manifold is more elongated and gives a more meaningful directions to use to interpret the result and make it consistent with the spatial projection

