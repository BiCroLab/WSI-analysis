#!/usr/bin/env python
import networkx as nx
import numpy as np
from graviti import *
import sys 
from scipy import sparse, linalg
import os
import seaborn as sns; sns.set()
import matplotlib
matplotlib.use('TKAgg',warn=False, force=True)
import matplotlib.pyplot as plt

filename = sys.argv[1] # name of the morphology mesearements from qupath
radius = int(sys.argv[2])   # for smoothing
quantiles = int(sys.argv[3]) # for stratifing the projection
threshold = int(sys.argv[4]) # min number of nodes in a subgraph

basename_graph = os.path.splitext(os.path.basename(filename))[0]
basename_smooth = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)+'.q'+str(quantiles)+'.t'+str(threshold)
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

####################################################################################################
# Select the morphological features,
# normalize the feature matrix
###################################################################################################

# Features list =  Nucleus:_Area   Nucleus:_Perimeter      Nucleus:_Circularity    Nucleus:_Eccentricity   Nucleus:_Hematoxylin_OD_mean    Nucleus:_Hematoxylin_OD_sum
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(7,8,9,12,13,14)).reshape((A.shape[0],6))

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
# Project principal components back to real space
###################################################################################################
print('Project back to real space')
import pandas as pd

node_colors = np.zeros(morphology.shape)
for f in range(morphology.shape[1]):
    node_colors[:,f] = pd.qcut(morphology_smooth[:,f].ravel(), quantiles, labels=False)

print('Done!')

####################################################################################################
# Get subgraphs from multi-features
###################################################################################################
print('Get the subgraphs')
outfile_subgraphs = os.path.join(dirname, basename)+'.subgraphs.npy'
outfile_node_list = os.path.join(dirname, basename)+'.node_list.npy'
if os.path.exists(outfile_subgraphs) and os.path.exists(outfile_node_list):
    subgraphs = np.load(outfile_subgraphs,allow_pickle=True)
    node_list = np.load(outfile_node_list,allow_pickle=True)
else:
    subgraphs, node_list = get_subgraphs(G,threshold,quantiles,node_colors)
    np.save(outfile_subgraphs,subgraphs)
    np.save(outfile_node_list,node_list)
print('Done!')

U = G.subgraph(node_list)
graphs = [g for g in list(nx.connected_component_subgraphs(U)) if g.number_of_nodes()>=threshold] # this threshold check might be redundant
# print('The list of nodes per graph is:')
# print([g.number_of_nodes() for g in graphs])

####################################################################################################
# Partition the graph and generate the covariance descriptors
# TO DO: substitute x,y with the modulus of the vector
###################################################################################################
print('Generate the covariance descriptor')
outfile_covd = os.path.join(dirname, basename)+'.covd.npy'
outfile_graph2covd = os.path.join(dirname, basename)+'.graph2covd.npy'
if os.path.exists(outfile_covd) and os.path.exists(outfile_graph2covd):
    covdata = np.load(outfile_covd,allow_pickle=True)
    graph2covd = np.load(outfile_graph2covd,allow_pickle=True)
else:
    # features = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(5,6,7,8,9,12,13,14)).reshape((A.shape[0],8)) #including X,Y
    features = np.hstack((pos,morphology_smooth)) # one should considere rotational invariant sqrt(x**2+y**2)
    covdata, graph2covd = covd_multifeature(features,G,subgraphs) # get list of cov matrices and a list of nodes per matrix
    np.save(outfile_covd,covdata)
    np.save(outfile_graph2covd,graph2covd)
print('Done!')
####################################################################################################
# Cluster the covariance descriptors
# TO DO: need to adjust clustering parameters to the statistics of the subgraph partitioning and try OPTICS instead of HDBSCAN
###################################################################################################
print('Cluster the descriptors')
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

logvec = [linalg.logm(m).reshape((1,16*16))  for m in covdata] #calculate the logm and vectorize
X = np.vstack(logvec) #create the array of covd data

standard_embedding = umap.UMAP(random_state=42).fit_transform(X) # this is used to plot
clusterable_embedding = umap.UMAP(n_neighbors=30,min_dist=0.0,n_components=2,random_state=42).fit_transform(X) # this is used to identify clusters

labels = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=50).fit_predict(clusterable_embedding)
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
for nodes in graph2covd: # for each subgraph
    node_cluster_color[nodes] = labels[ind] # update node_color with cluster labels for each node in the subgraph
    ind += 1

clustered = (node_cluster_color >= 0) # bolean array with true for clustered nodes and false for the rest
print(str(sum(clustered))+' nodes clustered of '+str(clustered.shape[0])+' in total')
clustered_nodes = np.asarray(list(G.nodes))[clustered] 
clustered_nodes_color = node_cluster_color[clustered] 
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

# ###################################################################################################
# # Prepare a new data set based on a given descriptor cluster
# ###################################################################################################
# clustered = (node_cluster_color >= 0)
# for cluster in set(node_cluster_color):
#     clustered = (node_cluster_color == cluster)
#     outfile = os.path.join(dirname, basename)+'.c'+str(int(cluster))
#     np.save(outfile,clustered)

