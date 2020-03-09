#!/usr/bin/env python
import networkx as nx
import numpy as np
from graviti import *
import sys 
from scipy import sparse, linalg
import os
import seaborn as sns; sns.set()

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

filename = sys.argv[1] # name of the morphology mesearements from qupath
radius = int(sys.argv[2])   # for smoothing
quantiles = int(sys.argv[3]) # for stratifing the projection
#threshold = int(sys.argv[4]) # min number of nodes in a subgraph

basename_graph = os.path.splitext(os.path.basename(filename))[0]
basename_smooth = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)+'.q'+str(quantiles)
elif os.path.splitext(os.path.basename(filename))[1] == '.txt':
    basename = os.path.splitext(os.path.basename(filename))[0]
dirname = os.path.dirname(filename)

####################################################################################################
# Construct the UMAP graph
# and save the adjacency matrix 
# and the degree and clustering coefficient vectors
###################################################################################################
print('Prepare the topological graph ...')
nn = 10 # this is hardcoded at the moment
path = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.adj.npz'
if not os.path.exists(path):
    print('The graph does not exists yet')
    A, pos = space2graph(filename,nn)
    sparse.save_npz(path, A)
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    d = getdegree(G)
    cc = clusteringCoeff(A)
    outfile = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.degree.gz'
    np.savetxt(outfile, d)
    outfile = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.cc.gz'
    np.savetxt(outfile, cc)
    nx.write_gpickle(G, os.path.join(dirname, basename_graph) + ".graph.pickle")
if os.path.exists(path):
    print('The graph exists already')
    A = sparse.load_npz(path) #id...graph.npz
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6))
    if os.path.exists( os.path.join(dirname, basename_graph) + ".graph.pickle" ):
        print('A networkx obj G exists already')
        G = nx.read_gpickle(os.path.join(dirname, basename_graph) + ".graph.pickle")
    else:
        print('A networkx obj G is being created')
        G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
        nx.write_gpickle(G, os.path.join(dirname, basename_graph) + ".graph.pickle")
    d = getdegree(G)
    cc = clusteringCoeff(A)
print('Topological graph ready!')
print('The graph has '+str(A.shape[0])+' nodes')
pos2norm = np.linalg.norm(pos,axis=1).reshape((pos.shape[0],1)) # the modulus of the position vector

####################################################################################################
# Select the morphological features,
# normalize the feature matrix
###################################################################################################

# Features list =  Nucleus:_Area   Nucleus:_Perimeter      Nucleus:_Circularity    Nucleus:_Eccentricity   Nucleus:_Hematoxylin_OD_mean    Nucleus:_Hematoxylin_OD_sum
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(7,8,9,12,13,14)).reshape((A.shape[0],6))
threshold = max(radius,(morphology.shape[1]+4)*2) # set the min subgraph size based on the dim of the feature matrix

####################################################################################################
# Smooth the morphology
###################################################################################################
print('Smooth the morphology')

outfile = os.path.join(dirname, basename)+'.smooth'
if os.path.exists(outfile+'.npy'):
    morphology_smooth = np.load(outfile+'.npy')
else:
    morphology_smooth = smoothing(A, morphology, radius)
    np.save(outfile, morphology_smooth)

print('Done!')

####################################################################################################
# Stratify morphologies
###################################################################################################
print('Stratify morphologies')
import pandas as pd

node_quantiles = np.zeros(morphology.shape)
for f in range(morphology.shape[1]):
    node_quantiles[:,f] = pd.qcut(morphology_smooth[:,f].ravel(), quantiles, labels=False)

print('Done!')

####################################################################################################
# Get subgraphs from multi-features as the connected components in each quantile
###################################################################################################
print('Get the subgraphs')
outfile_subgraphs = os.path.join(dirname, basename)+'.subgraphs.npy'
if os.path.exists(outfile_subgraphs):
    print('... loading the subgraphs ...')
    subgraphs = np.load(outfile_subgraphs,allow_pickle=True)
else:
    print('... creating the subgraphs ...')
    # subgraph above threshold with thresh large enough to filter part of the nodes otherwise the union graph will be the entire graph
    subgraphs, unique_nodes = get_subgraphs(G,threshold,quantiles,node_quantiles)
    
    np.save(outfile_subgraphs,subgraphs)
    np.save(outfile_nodes,unique_nodes)

print('There are '+str(len(subgraphs))+' subgraphs ')

print('Done')

####################################################################################################
# Generate the covariance descriptors
# this can be done with respect to raw features or smoothed ones
###################################################################################################
print('Generate the covariance descriptor')
features = np.hstack((pos2norm,morphology))            # this is rotational invariant

outfile_covd = os.path.join(dirname, basename)+'.covd.npy'
outfile_graph2covd = os.path.join(dirname, basename)+'.graph2covd.npy'
if os.path.exists(outfile_covd) and os.path.exists(outfile_graph2covd):
    print('... loading the descriptors ...')
    covdata = np.load(outfile_covd,allow_pickle=True)
    graph2covd = np.load(outfile_graph2covd,allow_pickle=True)
else:
    print('... creating the descriptors ...')
    covdata, graph2covd = covd_multifeature(features,G,subgraphs) # get list of cov matrices and a list of nodes per matrix
    np.save(outfile_covd,covdata)
    np.save(outfile_graph2covd,graph2covd)

print('There are '+str(len(covdata))+' covariance descriptors ')

####################################################################################################
# Cluster the covariance descriptors
###################################################################################################
print('Clustering the descriptors')
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

logvec = [linalg.logm(m).reshape((1,covdata[0].shape[0]*covdata[0].shape[1]))  for m in covdata] #calculate the logm and vectorize
X = np.vstack(logvec) #create the array of vectorized covd data

standard_embedding = umap.UMAP(random_state=42).fit_transform(X) # this is used to plot
clusterable_embedding = umap.UMAP(n_neighbors=30,min_dist=0.0,n_components=2,random_state=42).fit_transform(X) # this is used to identify clusters

# labels = hdbscan.HDBSCAN(min_samples=25,min_cluster_size=50).fit_predict(clusterable_embedding)
labels = OPTICS().fit(clusterable_embedding).labels_ # cluster label vector of covmatrices

clusteredD = (labels >= 0)
non_clusteredD = (labels < 0)
print(str(sum(clusteredD))+' descriptors clustered of '+str(labels.shape[0])+' in total')
print(str(sum(non_clusteredD))+' descriptors NOT clustered of '+str(labels.shape[0])+' in total')

plt.scatter(standard_embedding[~clusteredD, 0],
            standard_embedding[~clusteredD, 1],
            c='k',#(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(standard_embedding[clusteredD, 0],
            standard_embedding[clusteredD, 1],
            c=labels[clusteredD],
            s=0.1,
            cmap='viridis');

outfile = os.path.join(dirname, basename)+'.covd-clustering.png'
plt.savefig(outfile) # save as png
plt.close()
print('Done')

####################################################################################################
# Color nodes by labels
###################################################################################################

print('Color the graph by descriptor cluster')

node_cluster_color = -1.0*np.ones(A.shape[0]) #check
ind = 0
for nodes in graph2covd:                    # for each subgraph and corresponding covariance matrix
    node_cluster_color[nodes] = labels[ind] # update node_color with cluster labels for each node in the subgraph
    ind += 1

clusteredN = (node_cluster_color >= 0) # bolean array with true for clustered nodes and false for the rest
print(str(sum(clusteredN))+' nodes clustered of '+str(clusteredN.shape[0])+' in total')

non_clusteredN = (node_cluster_color < 0) # bolean array with true for clustered nodes and false for the rest
print(str(sum(non_clusteredN))+' nodes NOT clustered of '+str(clusteredN.shape[0])+' in total')

clustered_nodes = np.asarray(list(G.nodes))[clusteredN] 
clustered_nodes_color = node_cluster_color[clusteredN] 
subG = G.subgraph(clustered_nodes)

sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(subG, pos, alpha=0.5,node_color=clustered_nodes_color, node_size=1,cmap='viridis')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
outfile = os.path.join(dirname, basename)+'.node-clustering.png'
plt.savefig(outfile, dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()
print('Done!')

####################################################################################################
# Color nodes not clustered before
###################################################################################################
print('Determine the connected components of the non clustered nodes')

non_clustered_nodes = np.asarray(list(G.nodes))[non_clusteredN] 
subGnot = G.subgraph(non_clustered_nodes)
graphs = [g for g in list(nx.connected_component_subgraphs(subGnot)) if g.number_of_nodes()>=20]

print('Determine covariance matrix of the non clustered connected components')
covdata, graph2covd = covd_multifeature(features,G,graphs) 

logvec = [linalg.logm(m).reshape((1,covdata[0].shape[0]*covdata[0].shape[1])) for m in covdata] #calculate the logm and vectorize
X_nonclustered = np.vstack(logvec) #create the array of vectorized covd data

from scipy import spatial
reference = X[clusteredD,:]
tree = spatial.KDTree(reference)
for row_ind in range(X_nonclustered.shape[0]):
    index = tree.query(X_nonclustered[row_ind,:])[1]
    for nodes in graph2covd:                    
        node_cluster_color[nodes] = labels[index] 

sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_cluster_color, node_size=1,cmap='viridis')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
outfile = os.path.join(dirname, basename)+'.all-node-clustering.png'
plt.savefig(outfile, dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()

