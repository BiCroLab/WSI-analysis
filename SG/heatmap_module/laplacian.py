#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
import networkx as nx
warnings.filterwarnings('ignore')
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd
############################################
W = sparse.load_npz(sys.argv[1]) # adj.npz
npyfilename = sys.argv[2] # 'walkhistory.npy'
txtfilename = sys.argv[3] # 'data.txt.gz'
step = int(sys.argv[4]) # 0,1,2

cc = np.load(npyfilename,allow_pickle=True)[:,6,step] # 6 is the cc col num in the array
print(cc.shape)
S = normalize(W, norm='l1', axis=1) #create the row-stochastic matrix
L = nx.laplacian_matrix(nx.from_scipy_sparse_matrix(S,edge_attribute='weight')) #laplacian of the stochastic normalized W
print(L.shape)
attribute = L.dot(cc) # the node values of the Laplacian
####################################
# create empty list for node colors
G = nx.Graph()
pos = np.loadtxt(txtfilename, delimiter="\t",skiprows=True,usecols=(5,6))

G.add_nodes_from(range(len(attribute)))

#node_color = np.interp(attribute, (attribute.min(), attribute.max()), (0, +10))
node_color = pd.qcut(attribute, 10, labels=False)

# draw graph with node attribute color
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=1,cmap='viridis')

print('saving graph')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
plt.savefig(txtfilename+".laplacian.png", dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()

