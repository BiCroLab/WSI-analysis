#!/usr/bin/env python
# coding: utf-8

#################################
# Use the definition of clustering coefficient found in 
# https://content.iospress.com/articles/ai-communications/aic408
# (W^3)_kk / [(e^T.w_k)^2 - ||w_k||_2^2]
# where w_k is the k-th row of W
################################

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
import networkx as nx
warnings.filterwarnings('ignore')
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt

############################################
W = sparse.load_npz(sys.argv[1]) #load the graph matrix
npyfilename = sys.argv[2] #'/home/garner1/Work/pipelines/graviti/heatmap_module/npy/ID57-area-walkhistory.npy'
txtfilename = sys.argv[3] #'/home/garner1/Work/dataset/tissue2graph/ID57_data.txt'

step = int(sys.argv[4]) # which col in history

cc = np.load(npyfilename,allow_pickle=True)[:,step]

L = nx.laplacian_matrix(nx.from_scipy_sparse_matrix(W,edge_attribute='weight'))
print(L.shape)
attribute = L.dot(cc) # the node values of the Laplacian
print(attribute.shape)
####################################
# create empty list for node colors
G = nx.Graph()
pos = np.loadtxt(txtfilename, delimiter="\t",skiprows=True,usecols=(5,6))

G.add_nodes_from(range(len(attribute)))

node_color = np.interp(attribute, (attribute.min(), attribute.max()), (0, +10))

# draw graph with node attribute color
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=1,cmap='viridis')

print('saving graph')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
plt.savefig(txtfilename+".laplacian.png", dpi=200,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()

