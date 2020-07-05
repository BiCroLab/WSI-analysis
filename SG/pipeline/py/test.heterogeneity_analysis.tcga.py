#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys  
import glob
import os
sys.path.insert(0, '../py')
from graviti import *

def plot_loglog(df,title):
    values, bins = np.histogram(df['diversity'],bins=1000)
    y = values
    x = [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]

    plt.loglog(x, y,'r.')
    plt.xlabel("edge heterogeneity", fontsize=14)
    plt.ylabel("counts", fontsize=14)
    plt.title(title)
    plt.savefig(title+'.edgeH.loglog.png')
    plt.close()
    #plt.show()
    return
def plot_lognormal(df,title):
    values, bins = np.histogram(np.log2(df['diversity']),bins=100) # take the hist of the log values
    y = values
    x = [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]

    plt.plot(x, y,'r.')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel("Log_2 node heterogeneity", fontsize=14)
    plt.ylabel("counts", fontsize=14)
    plt.title(title)
    plt.savefig(title+'.nodeH.lognorm.png')
    plt.close()
    #plt.show()
    return


# In[2]:


for file in glob.glob('/home/garner1/wsi-data/heterogeneity_data/*.edge_diversity.tcga.pkl'):
    edges = pd.read_pickle(file)
    base=os.path.basename(file)
    for count in range(3):
        base = os.path.splitext(base)[0]
    plot_loglog(edges,base)
    N = 200
    contourPlot(edges[edges["diversity"]<20],N,np.median,base)


# In[5]:


for file in glob.glob('/home/garner1/wsi-data/heterogeneity_data/*.node_diversity.tcga.pkl'):
    base=os.path.basename(file)
    for count in range(3):
        base = os.path.splitext(base)[0]
    nodes = pd.read_pickle(file)
    plot_lognormal(nodes,base)


# In[ ]:




