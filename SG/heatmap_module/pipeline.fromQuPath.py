#!/usr/bin/env python
import networkx as nx
import numpy as np
from graviti import *
import sys 
from scipy import sparse
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

filename = sys.argv[1] # name of the morphology mesearements from qupath
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]
elif os.path.splitext(os.path.basename(filename))[1] == '.txt':
    basename = os.path.splitext(os.path.basename(filename))[0]
dirname = os.path.dirname(filename)

####################################################################################################
# Construct the UMAP graph
# and save the adjacency matrix 
# and the degree and clustering coefficient vectors
###################################################################################################

A, pos, nn = space2graph(filename)
G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
d = getdegree(G)
cc = clusteringCoeff(A)

outfile = os.path.join(dirname, basename)+'.nn'+str(nn)+'.adj.npz'
sparse.save_npz(outfile, A)
outfile = os.path.join(dirname, basename)+'.nn'+str(nn)+'.degree.gz'
np.savetxt(outfile, d)
outfile = os.path.join(dirname, basename)+'.nn'+str(nn)+'.cc.gz'
np.savetxt(outfile, cc)

####################################################################################################
# Select the morphological features,
# normalize the feature matrix
# and perform PCA analysis
###################################################################################################

# Features list =  Nucleus:_Area   Nucleus:_Perimeter      Nucleus:_Circularity    Nucleus:_Eccentricity   Nucleus:_Hematoxylin_OD_mean    Nucleus:_Hematoxylin_OD_sum
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(7,8,9,12,13,14)).reshape((A.shape[0],6))
morphology_scaled = rescale(morphology)

####################################################################################################
# Perform PCA analysis
###################################################################################################
import pickle as pk

pca = principalComp(morphology_scaled)
outfile = os.path.join(dirname, basename)+".pca.pkl"
pk.dump(pca, open(outfile,"wb"))

# print(pca.explained_variance_ratio_)
# print(pca.n_components_,pca.n_features_,pca.n_samples_)
# print(pca.singular_values_)
# print(pca.components_)

####################################################################################################
# Project principal components back to real space
###################################################################################################
projection = np.dot(morphology_scaled,pca.components_.transpose())
booleanProj = (projection > 0)

uniqueValues, occurCount = np.unique(booleanProj, axis=0, return_counts=True)

listOfUniqueValues = zip(uniqueValues, range(uniqueValues.shape[0]))
print('Unique Values along with their cluster id')
positions = [] # a list of list of indexes
node_color = np.zeros((booleanProj.shape[0],1))
for elem in listOfUniqueValues:
   newlist = np.where(np.all(booleanProj==elem[0],axis=1))[0].tolist() #row index where a cluster occur
   node_color[newlist] = elem[1]
   positions.append(newlist)

node_color = node_color.flatten() # to be used in colormap 

####################################################################################################
# Draw graph with node attribute color
###################################################################################################
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=1,cmap='viridis')

print('saving graph')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
outfile = os.path.join(dirname, basename)+'.heatmap'
plt.savefig(outfile+".png", dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()
