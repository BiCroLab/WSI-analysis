#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy import sparse
import sys
import umap
import warnings
import networkx as nx
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, zscore, poisson
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

txtfilename = sys.argv[1] #'txt.gz'
datafilename = sys.argv[2] # 'walkhistory.npy' or 'smooth.py'
pcfilename = sys.argv[3]
component = int(sys.argv[4])
time = int(sys.argv[5])

data = np.load(datafilename,allow_pickle=True)[:,:,time]
data = normalize(data, norm='l1', axis=0) #create the col-stochastic matrix

pc = np.load(pcfilename,allow_pickle=True)[component,:]

projection = np.abs(np.dot(data,pc.transpose()))

# create empty list for node colors
G = nx.Graph()
pos = np.loadtxt(txtfilename, delimiter="\t",skiprows=True,usecols=(5,6))
G.add_nodes_from(range(projection.shape[0]))

# color attribute based on percentiles, deciles or quartiles ...
#if modality == 'absolute':
#node_color = np.interp(projection, (projection.min(), projection.max()), (0, +10))
#elif modality == 'deciles':
node_color = pd.qcut(projection, 10, labels=False)

# draw graph with node attribute color
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=1,cmap='viridis')

print('saving graph')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
plt.savefig(txtfilename+".component-"+str(component)+".window-"+str(time)+".png", dpi=100,bbox_inches = 'tight', pad_inches = 0.5) 
plt.close()

