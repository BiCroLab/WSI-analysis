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

import warnings
warnings.filterwarnings("ignore")

import plotly
plotly.io.orca.config.executable = '/usr/local/share/anaconda3/bin/orca' # this has to be hard-coded for each machine

# Read the full dataframe with cluster IDs
df = pd.read_csv('/home/garner1/Work/pipelines/WSI-analysis/SG/pipeline/pkl/id_52.fov_centroids_embedding_morphology.covd.pkl.size900356.intensity.csv.gz').sample(n=300000)

from sklearn import preprocessing
# rescale the features to 0-1 range
for feature in ["area","perimeter","solidity","eccentricity"]:
    f = df[feature].as_matrix().reshape(-1,1) #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    f_scaled = min_max_scaler.fit_transform(f)
    df[feature] = f_scaled

min_cluster_size = 50000 # parameter to be adjausted
min_samples = 1000      # parameter to be adjausted
clusterer = hdbscan.HDBSCAN(min_samples=min_samples,
                            min_cluster_size=min_cluster_size,
                            gen_min_span_tree=True)
clusterer.fit( df[["area","perimeter","solidity","eccentricity"]] ) 
clusterer.condensed_tree_.plot(select_clusters=True,
                               selection_palette=sns.color_palette("Set2",len(clusterer.labels_)))
plt.savefig('test.png')

#update datarrame
df["cluster_morphology"] = clusterer.labels_
print(set(df["cluster_morphology"] ))
