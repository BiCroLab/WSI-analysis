#!/usr/bin/env python
import networkx as nx
import numpy as np
from graviti import *
import sys 
from scipy import sparse, linalg
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

filename = sys.argv[1] # name of the morphology mesearements from qupath
clusterfile = sys.argv[2] # name of the npy cluster boolean vec
radius = int(sys.argv[3])   # for smoothing
quantiles = int(sys.argv[4]) # for stratifing the projection
threshold = int(sys.argv[5]) # min number of nodes in a subgraph

basename_graph = os.path.splitext(os.path.basename(filename))[0]
basename_smooth = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)
basename = os.path.splitext(os.path.basename(clusterfile))[0]+'.r'+str(radius)+'.q'+str(quantiles)+'.t'+str(threshold)
dirname = os.path.dirname(filename)
cluster = np.load(clusterfile)
####################################################################################################
# Construct the UMAP graph
# and save the adjacency matrix 
# and the degree and clustering coefficient vectors
###################################################################################################
print('Prepare the topological graph ...')
nn = 10 # this is hardcoded at the moment
path = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.adj.npz'
pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6))
nodes = [i for i, x in enumerate(cluster) if x]
G = nx.read_gpickle(os.path.join(dirname, basename_graph) + ".graph.pickle").subgraph(nodes)
A = nx.to_scipy_sparse_matrix(G)
# d = getdegree(G)
# cc = clusteringCoeff(A)
print('Topological graph ready!')

####################################################################################################
# Select the morphological features,
# normalize the feature matrix
###################################################################################################

# Features list =  Nucleus:_Area   Nucleus:_Perimeter      Nucleus:_Circularity    Nucleus:_Eccentricity   Nucleus:_Hematoxylin_OD_mean    Nucleus:_Hematoxylin_OD_sum
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(7,8,9,12,13,14))[cluster,:].reshape((sum(cluster),6))

####################################################################################################
# Smooth the morphology
###################################################################################################
print('Smooth the morphology')

outfile = os.path.join(dirname, basename_smooth)+'.smooth'
morphology_smooth = np.load(outfile+'.npy')[cluster,:]

print('Done!')

####################################################################################################
# Perform PCA analysis
###################################################################################################
print('Run PCA')
import pickle as pk
morphology_scaled = rescale(morphology_smooth) # into [-1,+1] per feature
pca = principalComp(morphology_scaled)
outfile = os.path.join(dirname, basename)+".pca.pkl"
pk.dump(pca, open(outfile,"wb"))
print('Done!')

####################################################################################################
# Project principal components back to real space
###################################################################################################
print('Project back to real space')
import pandas as pd

projection = np.dot(morphology_scaled,pca.components_.transpose()[:,0]) #project only the first PC
node_color = pd.qcut(projection, quantiles, labels=False)
print('Done!')

####################################################################################################
# Partition the graph and generate the covariance descriptors
###################################################################################################
print('Generate the covariance descriptor')
outfile_covd = os.path.join(dirname, basename)+'.covd.npy'
outfile_graph2covd = os.path.join(dirname, basename)+'.graph2covd.npy'
if os.path.exists(outfile_covd) and os.path.exists(outfile_graph2covd):
    covdata = np.load(outfile_covd,allow_pickle=True)
    graph2covd = np.load(outfile_graph2covd,allow_pickle=True)
else:
    features = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(5,6,7,8,9,12,13,14))[cluster,:].reshape((sum(cluster),8)) #including X,Y
    covdata, graph2covd = covd(features,G,threshold,quantiles,node_color)
    np.save(outfile_covd,covdata)
    np.save(outfile_graph2covd,graph2covd)
print('Done!')

####################################################################################################
# Cluster the covariance descriptors
###################################################################################################
print('Cluster the descriptors')
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

flat_covdata = [item for sublist in covdata for item in sublist] #flatten the list of lists
logvec = [linalg.logm(m).reshape((1,16*16))  for m in flat_covdata] #calculate the logm and vectorize
X = np.vstack(logvec) #create the array of covd data

standard_embedding = umap.UMAP(random_state=42).fit_transform(X)
clusterable_embedding = umap.UMAP(n_neighbors=10,min_dist=0.0,n_components=2,random_state=42).fit_transform(X)

labels = hdbscan.HDBSCAN(min_samples=5,min_cluster_size=10).fit_predict(clusterable_embedding)
clustered = (labels >= 0)
plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            c='k',#(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral')
outfile = os.path.join(dirname, basename)+'.covd-clustering.png'
plt.savefig(outfile) # save as png
plt.close()
print('Done')

####################################################################################################
# Color nodes by labels
###################################################################################################
if sum(clustered) > 0:
    print('Color the graph by descriptor cluster')
    ind = 0
    node_cluster_color = -1*np.ones(node_color.shape)
    for item in graph2covd: # for each subgraph
        node_cluster_color[list(item[0][1])] = labels[ind] # update node_color with cluster labels for each node in the subgraph
        ind += 1

    clustered = (node_cluster_color >= 0)
    clustered_nodes = np.asarray(list(G.nodes))[clustered] 
    clustered_nodes_color = node_cluster_color[clustered] 
    subG = G.subgraph(clustered_nodes)

    sns.set(style='white', rc={'figure.figsize':(50,50)})
    nx.draw_networkx_nodes(subG, pos, alpha=0.5, node_color=clustered_nodes_color, node_size=1, cmap='viridis')
    
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    plt.axis('off')
    outfile = os.path.join(dirname, basename)+'.node-clustering.png'
    plt.savefig(outfile, dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
    plt.close()
    print('Done!')

