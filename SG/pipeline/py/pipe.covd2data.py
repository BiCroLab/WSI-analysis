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

def filtering(df):
    #First removing columns
    filt_df = df[["area","perimeter","solidity","eccentricity","circularity","mean_intensity","std_intensity"]]
    df_keep = df.drop(["area","perimeter","solidity","eccentricity","circularity","mean_intensity","std_intensity"], axis=1)
    #Then, computing percentiles
    low = .01
    high = .99
    quant_df = filt_df.quantile([low, high])
    #Next filtering values based on computed percentiles
    filt_df = filt_df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                        (x < quant_df.loc[high,x.name])], axis=0)
    #Bringing the columns back
    filt_df = pd.concat( [df_keep,filt_df], axis=1 )
    filt_df['cov_intensity'] = filt_df['std_intensity']/filt_df['mean_intensity']
    #rows with NaN values can be dropped simply like this
    filt_df.dropna(inplace=True)
    return filt_df

dirname = sys.argv[1] # the directory where covd.npz files are located
sample = sys.argv[2]  # the sample id

counter = 0
for f in glob.glob(dirname+'/*covd.npz'): # for every fov
    counter += 1
    if counter == 1:            # to set up the data arrays
        data = np.load(f,allow_pickle=True)
        fov = data['fov']
        covds = data['descriptors']
        xy = data['centroids']
        morphology = data['morphology']
    else:                       # to update the data arrays
        data = np.load(f,allow_pickle=True)
        fov = np.vstack((fov,data['fov']))
        covds = np.vstack((covds, data['descriptors']))
        xy = np.vstack((xy, data['centroids']))
        morphology = np.vstack((morphology, data['morphology']))

# Create dataframes
df_fov = pd.DataFrame(data=fov, columns=['fov_row','fov_col'])
df_xy = pd.DataFrame(data=xy, columns=['cx','cy'])
df_morphology = pd.DataFrame(data=morphology, columns=['area','perimeter','solidity','eccentricity','circularity','mean_intensity','std_intensity'])
df_covds = pd.DataFrame(data=covds)

# Concatenate all dataframes
df = pd.concat([df_fov,df_xy, df_morphology, df_covds],axis=1)

# filter by percentiles in morphologies (hardcoded in function filtering)
fdf = filtering(df)

# UMAP representation of the intensity descriptors, always check the proper column selection!
embedding_intensity = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(fdf[fdf.columns[4:-8]]) 

# Create dataframes of the umap embedding
df_embedding_intensity = pd.DataFrame(data=embedding_intensity, columns=['xi','yi','zi'])

# Concatenate the embedded dataframes
fdf.reset_index(drop=True, inplace=True)
df_embedding_intensity.reset_index(drop=True, inplace=True)
df_final = pd.concat([fdf[fdf.columns[:4]], fdf[fdf.columns[-8:]], df_embedding_intensity],axis=1)

# Save dataframe
df_final.to_pickle("id_"+str(sample)+".measurements.covd.pkl")

