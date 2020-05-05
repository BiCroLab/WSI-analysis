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
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

import os
import glob

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
from sklearn.metrics import pairwise_distances_argmin_min
import hdbscan

#def cluster_nuclei(filename,sample_size,n_neighbors,threshold_q,min_cluster_size,min_samples):
def cluster_nuclei_intensity(filename,sample_size,n_neighbors,threshold_q,auto_open,plot_switch):
    df = pd.read_pickle(filename)
    if sample_size > 0 and sample_size < df.shape[0]:
        df = df.sample(n=sample_size)
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
    df1['cluster_intensity'] = df1['cluster_intensity'].apply(str)   # make cluster id a string
    df1_filtered = df1[df1.cluster != str(-1)] # remove unassigned points

    # expand the clusters to the entire point-cloud
    idx, dist = pairwise_distances_argmin_min(df[['xi','yi','zi']].to_numpy(),df1_filtered[['xi','yi','zi']].to_numpy())
    df['cluster_intensity'] = [int(df1_filtered.cluster_intensity.iloc[idx[row]])+1 for row in range(df.shape[0])] #add 1 to avoid confusion with background
    df.to_csv(filename+'.intensity.csv',index=False)
    
    if plot_switch:
        # plot the spatial projetion
        fig = px.scatter(df1_filtered, x="cx", y="cy",color="cluster_intensity",
                             width=800, height=800,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker=dict(size=5,opacity=1.0))
        fig.write_html(filename+'.spatial_projection.intensity.html', auto_open=auto_open)
        fig.write_image(filename+'.spatial_projection.intensity.png')

        # plot the low curvature sector
        fig = px.scatter_3d(df1_filtered, x="xi", y="yi", z="zi", 
                            color="cluster_intensity", hover_name="cluster_intensity",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker=dict(size=3,opacity=0.75),selector=dict(mode='markers'))
        fig.write_html(filename+'.low_curvature_clusters.intensity.html', auto_open=auto_open)
        fig.write_image(filename+'.low_curvature_clusters.intensity.png')
    return df

#def cluster_nuclei(filename,sample_size,n_neighbors,threshold_q,min_cluster_size,min_samples):
def cluster_nuclei_morphology(filename,sample_size,n_neighbors,threshold_q,auto_open,plot_switch):
    df = pd.read_pickle(filename)
    if sample_size > 0 and sample_size < df.shape[0]:
        df = df.sample(n=sample_size)
    embedding = df[['xm','ym','zm']].to_numpy()
    
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
    clusterer.fit(df1.loc[:,('xm','ym','zm')]) 

    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=sns.color_palette("Set2",len(clusterer.labels_)))
    plt.savefig(filename+'.tree.morphology.png')
    plt.close()
    
    df1['cluster_morphology'] = clusterer.labels_    # add cluster id to dataframe
    df1['cluster_morphology'] = df1['cluster_morphology'].apply(str)   # make cluster id a string
    df1_filtered = df1[df1.cluster != str(-1)] # remove unassigned points

    # expand the clusters to the entire point-cloud
    idx, dist = pairwise_distances_argmin_min(df[['xm','ym','zm']].to_numpy(),df1_filtered[['xm','ym','zm']].to_numpy())
    df['cluster_morphology'] = [int(df1_filtered.cluster_morphology.iloc[idx[row]])+1 for row in range(df.shape[0])] #add 1 to avoid confusion with background
    df.to_csv(filename+'.morphology.csv',index=False)
    
    if plot_switch:
        # plot the spatial projetion
        fig = px.scatter(df1_filtered, x="cx", y="cy",color="cluster_morphology",
                             width=800, height=800,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker=dict(size=5,opacity=1.0))
        fig.write_html(filename+'.spatial_projection.morphology.html', auto_open=auto_open)
        fig.write_image(filename+'.spatial_projection.morphology.png')

        # plot the low curvature sector
        fig = px.scatter_3d(df1_filtered, x="xm", y="ym", z="zm", 
                            color="cluster_morphology", hover_name="cluster_morphology",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker=dict(size=3,opacity=0.75),selector=dict(mode='markers'))
        fig.write_html(filename+'.low_curvature_clusters.morphology.html', auto_open=auto_open)
        fig.write_image(filename+'.low_curvature_clusters.morphology.png')
    return df

sample_size = 10000 # set to 0 if the entire sample is considered
n_neighbors = 100   # NNN in the curvature calculation
threshold_q = 0.1   # the quantile defining the low-curvature sector
auto_open = True    # switch to open or not html figures in new tab
plot_switch = True  # switch to generate or not html figures
#min_cluster_size = 1000   # in hdbscan 
#min_samples = 500         # in hdbscan

for filename in glob.glob(r'../pkl/id_13.*.pkl'):
    #df_out = cluster_nuclei(filename,sample_size,n_neighbors,threshold_q,min_cluster_size,min_samples)
    df_intensity = cluster_nuclei_intensity(filename,sample_size,n_neighbors,threshold_q,
                                       auto_open=auto_open,plot_switch=plot_switch)
    df_morphology = cluster_nuclei_morphology(filename,sample_size,n_neighbors,threshold_q,
                                       auto_open=auto_open,plot_switch=plot_switch)

