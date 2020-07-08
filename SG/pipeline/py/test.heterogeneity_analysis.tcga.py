#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys  
import glob
import os
sys.path.insert(0, '../py')
from graviti import *



# # In[2]:

# for filename in glob.glob('/home/garner1/Work/pipelines/WSI-analysis/SG/pipeline/pkl/*.edge_diversity.tcga.pkl'):
#     print(filename)
#     edges = pd.read_pickle(filename)
#     base=os.path.basename(filename)
#     for count in range(3):
#         base = os.path.splitext(base)[0]
#     # plot_loglog(edges,base)
#     N = 200
#     contourPlot(edges[edges["diversity"]<20.0],N,np.sum,base)


# # In[5]:

# for filename in glob.glob('/home/garner1/Work/pipelines/WSI-analysis/SG/pipeline/pkl/*.node_diversity.tcga.pkl'):
#     print(filename)
#     base=os.path.basename(filename)
#     for count in range(3):
#         base = os.path.splitext(base)[0]
#     nodes = pd.read_pickle(filename)
#     plot_lognormal(nodes,base)

samples = glob.glob('/home/garner1/Work/pipelines/WSI-analysis/SG/pipeline/pkl/*.node_diversity.tcga.pkl')
p_arr = np.zeros((len(samples),17))

sample_count = 0
for filename in samples:
    print(filename)
    base=os.path.basename(filename)
    for count in range(3):
        base = os.path.splitext(base)[0]
    nodes = pd.read_pickle(filename)

    feature_count = 0
    for feat in nodes.columns[7:-2]:
        pearson = nodes[feat].corr(nodes['diversity'],method='pearson')
        p_arr[sample_count,feature_count] = pearson
        feature_count += 1
        
        #nodes.plot(x=feat, y='diversity', style='o',title=str(pearson))
        # plt.savefig(feat+'-diversity.png')
        # print(feat,nodes[feat].corr(nodes['diversity'],method='pearson'))
    sample_count += 1

p_df = pd.DataFrame(p_arr,columns=nodes.columns[7:-2])
hist = p_df.hist(bins=10,xlabelsize=8,ylabelsize=5,figsize=(20, 20))
[x.title.set_size(10) for x in hist.ravel()]
plt.savefig('tcga.pearson_histograms.png')

