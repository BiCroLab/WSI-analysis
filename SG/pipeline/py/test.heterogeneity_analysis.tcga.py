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



# In[2]:

for filename in glob.glob('/home/garner1/Work/pipelines/WSI-analysis/SG/pipeline/pkl/*.edge_diversity.tcga.pkl'):
    print(filename)
    edges = pd.read_pickle(filename)
    base=os.path.basename(filename)
    for count in range(3):
        base = os.path.splitext(base)[0]
    # plot_loglog(edges,base)
    N = 200
    contourPlot(edges[edges["diversity"]<20.0],N,np.sum,base)


# In[5]:


# for filename in glob.glob('/home/garner1/Work/pipelines/WSI-analysis/SG/pipeline/pkl/*.node_diversity.tcga.pkl'):
#     print(filename)
#     base=os.path.basename(filename)
#     for count in range(3):
#         base = os.path.splitext(base)[0]
#     nodes = pd.read_pickle(filename)
#     plot_lognormal(nodes,base)


# In[ ]:




