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
import warnings
warnings.filterwarnings('ignore')

dirname = sys.argv[1] # the directory where covd.npz files are located
sample = sys.argv[2]  # the sample id
counter = 0
for f in glob.glob(dirname+'/*covd.npz'):
    counter += 1
    if counter == 1:
        data = np.load(f,allow_pickle=True)
        fov = data['fov']
        covds = data['descriptors']
        xy = data['centroids']
        morphology = data['morphology']
    else:
        data = np.load(f,allow_pickle=True)
        fov = np.vstack((fov,data['fov']))
        covds = np.vstack((covds, data['descriptors']))
        xy = np.vstack((xy, data['centroids']))
        morphology = np.vstack((morphology, data['morphology']))

print('Clustering the descriptors')
embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(covds) 
df_fov = pd.DataFrame(data=fov, columns=['fov_row','fov_col'])
df_embedding = pd.DataFrame(data=embedding, columns=['x','y','z'])
df_xy = pd.DataFrame(data=xy, columns=['cx','cy'])
df_morphology = pd.DataFrame(data=morphology, columns=['area','perimeter','solidity','eccentricity','mean_intensity'])
df = pd.concat([df_fov,df_xy, df_embedding, df_morphology],axis=1)
print(df)
df.to_pickle("id_"+str(sample)+".fov_centroids_embedding_morphology.covd.pkl") # save datafram

