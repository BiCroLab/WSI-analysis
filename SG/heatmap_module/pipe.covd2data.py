#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from skimage.measure import label, regionprops
from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
from scipy.linalg import eigh, inv, logm, norm
from  scipy import ndimage,sparse
import cv2
import os
import sys
import csv
import glob

import h5py
from matplotlib import pyplot as plt
import warnings
#warnings.filterwarnings('ignore')


# In[ ]:


'''
Set the input information
'''
h5_file = sys.argv[1]   #this file contains the segmented nuclei
datadir = os.path.dirname(os.path.realpath(h5_file))
dapi_file = sys.argv[2] #this file contains the tif images
npz_file = sys.argv[3] #this is the output file with spatial and morphological descriptors
method = sys.argv[4] #choose between covd rotational invariant or not: covdRI or covd 
report = sys.argv[5] #filename of the output report


# In[ ]:


dirname = sys.argv[1]
counter = 0
for f in glob.glob(dirname+'/*.npz'):
    if counter == 0:
        data = np.load(f,allow_pickle=True) 
        covds = data['descriptors']
    data = np.load(f,allow_pickle=True)
    covds = np.vstack((covds,data['descriptors']))


# In[ ]:


print('Clustering the descriptors')
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.cluster import OPTICS

# this is used to identify clusters                                 
embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(covds) 


# In[ ]:


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

import hdbscan

import os
import glob

from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
import numpy as np
import pandas as pd


# In[ ]:


df_embedding = pd.DataFrame(data=embedding, columns=['x','y','z'])
'''
Visualize the 3D UMAP representation of the morphology
'''
fig = px.scatter_3d(df_embedding, x="x", y="y", z="z")
fig.update_traces(marker=dict(size=1,opacity=0.5),selector=dict(mode='markers'))
fig.write_html('test.html', auto_open=True)


# In[ ]:




