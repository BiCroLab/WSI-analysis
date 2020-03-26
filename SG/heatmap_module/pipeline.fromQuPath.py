#!/usr/bin/env python
import networkx as nx
import numpy as np
from graviti import *
import sys 
from scipy import sparse, linalg
import os
import copy
import seaborn as sns; sns.set()

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

filename = sys.argv[1] # name of the morphology mesearements from qupath
radius = int(sys.argv[2])   # for smoothing
rndsample = int(sys.argv[3])    # for community detection stochasticity

basename_graph = os.path.splitext(os.path.basename(filename))[0]
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)+'.s'+str(rndsample)
dirname = os.path.dirname(filename)

####################################################################################################
# Construct the UMAP graph
# and save the adjacency matrix 
# and the degree and clustering coefficient vectors
###################################################################################################
print('Prepare the topological graph ...')
nn = 10 # this is hardcoded at the moment
path = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.adj.npz'

if os.path.exists(path) and os.path.exists( os.path.join(dirname, basename_graph) + ".graph.pickle" ):
    print('The graph exists already')
    A = sparse.load_npz(path) #id...graph.npz
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6)) # here there is no header
    G = nx.read_gpickle(os.path.join(dirname, basename_graph) + ".graph.pickle")
    d = getdegree(G)
    cc = clusteringCoeff(A)
else:
    print('The graph does not exists yet')
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6)) # here there is no header
    A = space2graph(pos,nn)
    sparse.save_npz(path, A)
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    d = getdegree(G)
    cc = clusteringCoeff(A)
    outfile = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.degree.gz'
    np.savetxt(outfile, d)
    outfile = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.cc.gz'
    np.savetxt(outfile, cc)
    nx.write_gpickle(G, os.path.join(dirname, basename_graph) + ".graph.pickle")

pos2norm = np.linalg.norm(pos,axis=1).reshape((pos.shape[0],1)) # the modulus of the position vector to preserve translational invariance

print('Topological graph ready!')
print('The graph has '+str(A.shape[0])+' nodes')

####################################################################################################
# Select the morphological features,
# and set the min number of nodes per subgraph
###################################################################################################
print('Prepare the morphology array')
# Features list =  Nucleus:_Area   Nucleus:_Perimeter      Nucleus:_Circularity    Nucleus:_Eccentricity   Nucleus:_Hematoxylin_OD_mean    Nucleus:_Hematoxylin_OD_sum
from sklearn.preprocessing import normalize 
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(7,8,9,12,13,14)).reshape((A.shape[0],6))
threshold = max(radius,(morphology.shape[1]+4)*2) # set the min subgraph size based on the dim of the feature matrix

####################################################################################################
# Smooth the morphology
###################################################################################################
print('Smooth the morphology')

outfile = os.path.join(dirname, basename)+'.smooth'
if os.path.exists(outfile+'.npy'):
    morphology_smooth = np.load(outfile+'.npy', allow_pickle=True)
else:
    morphology_smooth = smoothing(A, morphology, radius)
    np.save(outfile, morphology_smooth)

####################################################################################################
# Reweight the graph
###################################################################################################
print('Rescale graph weights by local morphology')

print('...use the smooth morphology...')
# morphology_normed = normalize(morphology, norm='l1', axis=0) # normalize features
morphology_normed = normalize(morphology_smooth, norm='l1', axis=0) # normalize features
GG = copy.deepcopy(G)
for ijw in G.edges(data='weight'):
    feature = np.asarray([ abs(morphology_normed[ijw[0],f]-morphology_normed[ijw[1],f]) for f in range(morphology_normed.shape[1]) ]) # array of morphology features 
    GG[ijw[0]][ijw[1]]['weight'] = ijw[2]/np.sum(feature) # the new graph weights is the ratio between the current and the sum of the neightbours differences

####################################################################################################
# Community detection
###################################################################################################
print('Find the communities in GG')

from cdlib import algorithms
from cdlib import evaluation
from cdlib.utils import convert_graph_formats
import igraph
import leidenalg
from networkx.algorithms.community.quality import modularity

print('...convert networkx graph to igraph object...')
g = igraph.Graph(directed=False)
g.add_vertices(list(GG.nodes()))
g.add_edges(list(GG.edges()))
edgelist = nx.to_pandas_edgelist(GG)
for attr in edgelist.columns[2:]:
    g.es[attr] = edgelist[attr]

print('...finding the partitions...')
outfile_communities = os.path.join(dirname, basename)+'.communities.npy'
if os.path.exists(outfile_communities):
    communities = np.load(outfile_communities,allow_pickle=True)
else:
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,initial_membership=None, weights='weight', seed=rndsample, n_iterations=2)
    communities = [g.vs[x]['name'] for x in part]
    np.save(outfile_communities, communities)

partition = [set(map(int, set(sg))) for sg in communities] # make a list of set of partitions
mod_GG = modularity(GG, partition, weight='weight')
mod_G = modularity(G, partition, weight='weight')

print('The modularity of GG is '+str(mod_GG))
print('The modularity of G is '+str(mod_G))
    
bigcommunities = [g for g in communities if len(g) > threshold] # list of big enough communities
outfile = os.path.join(dirname, basename)+'.bigcommunities'
np.save(outfile, bigcommunities)

print('There are '+str(len(bigcommunities))+' big communities and '+str(len(communities))+' communities in total')

####################################################################################################
# Generate the covariance descriptors
# this can be done with respect to raw features or smoothed ones
###################################################################################################
print('Generate the covariance descriptor')
features = np.hstack((pos2norm,morphology))            # this is rotational invariant

outfile_covd = os.path.join(dirname, basename)+'.covd.npy'
if os.path.exists(outfile_covd):
    print('... loading the descriptors ...')
    covdata = np.load(outfile_covd,allow_pickle=True)
else:
    print('... creating the descriptors ...')
    covdata = community_covd(features,G,bigcommunities) # get list of cov matrices and a list of nodes per matrix
    np.save(outfile_covd,covdata)

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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

print('...prepare the data...')
outfile_logvec = os.path.join(dirname, basename)+'.logvec.npy'
if os.path.exists(outfile_logvec):
    print('...load the logvec dataset...')
    X = np.load(outfile_logvec,allow_pickle=True)
else:
    print('...create the logvec dataset...')
    logvec = [linalg.logm(m).reshape((1,covdata[0].shape[0]*covdata[0].shape[1]))  for m in covdata] #calculate the logm and vectorize
    X = np.vstack(logvec) #create the array of vectorized covd data
    np.save(outfile_logvec,X)
print('The vectorized covd array has shape '+str(X.shape))    
outfile_clusterable_embedding = os.path.join(dirname, basename)+'.clusterable_embedding.npy'
if os.path.exists(outfile_clusterable_embedding):
    print('...load the clusterable embedding...')
    clusterable_embedding = np.load(outfile_clusterable_embedding,allow_pickle=True)
else:
    print('...create the clusterable embedding...')
    clusterable_embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(X) # this is used to identify clusters
    np.save(outfile_clusterable_embedding,clusterable_embedding)

print('The embedding has shape '+str(clusterable_embedding.shape))

####################################################################################################
# Free up spaces
###################################################################################################
del G                           # G is not needed anymore
del A                           # A is not needed anymore
del morphology
del morphology_smooth

####################################################################################################
# Color graph nodes by community label
###################################################################################################
print('Preparing to color the graph communities')
print('...set up the empty graph...')
g = nx.Graph()
g.add_nodes_from(range(pos.shape[0])) # add all the nodes of the graph, but not all of them are in a covd cluster because of small communities
print('...set up the empty dictionary...')
dictionary = {}
for node in range(pos.shape[0]):
    dictionary[int(node)] = -1 # set all node to -1

print('...set up the full dictionary...')
node_comm_tuples = [(int(node),i) for i, community in enumerate(bigcommunities) for node in community]
dictionary.update(dict(node_comm_tuples))

node_color = []
for i in sorted (dictionary) :  # determine the color based on the community
    node_color.append(dictionary[i])

print('...draw the graph...')
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(g, pos, alpha=0.5,node_color=node_color, node_size=1,cmap=plt.cm.Set1)

print('...saving graph...')
plt.axis('off')
plt.savefig(os.path.join(dirname, basename)+'.community_graph.png') # save as png
plt.close()

###################################################################################################
###################################################################################################

# # print('...cluster the descriptors...')
# # labels = hdbscan.HDBSCAN(min_samples=50,min_cluster_size=100).fit_predict(clusterable_embedding)
# labels = OPTICS(min_samples=50, xi=.01, min_cluster_size=.05).fit_predict(clusterable_embedding) # cluster label vector of covmatrices
# # labels = AgglomerativeClustering(n_clusters=4).fit_predict(clusterable_embedding)
# # labels = SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=42).fit_predict(clusterable_embedding) # mem demanding

# outfile_cluster_labels = os.path.join(dirname, basename)+'.cluster_labels.npy'
# np.save(outfile_cluster_labels, labels)
# print('There are '+str(len(set(labels)))+' clusters:'+str(set(labels)))
# cluster_features = cluster_morphology(morphology_smooth,graph2covd,labels)
# outfile_cluster_features = os.path.join(dirname, basename)+'.cluster_features.npy'
# print('The shape of the cluster feature matrix is ',str(cluster_features.shape))
# np.save(outfile_cluster_features, cluster_features)

# print('...plot the clusters...')
# clusteredD = (labels >= 0)
# non_clusteredD = (labels < 0)
# print(str(sum(clusteredD))+' descriptors clustered of '+str(labels.shape[0])+' in total')
# print(str(sum(non_clusteredD))+' descriptors NOT clustered of '+str(labels.shape[0])+' in total')

# plt.scatter(clusterable_embedding[:, 0],
#             clusterable_embedding[:, 1],
#             c='k',#(0.5, 0.5, 0.5),
#             s=0.1,
#             alpha=0.5)

# plt.scatter(clusterable_embedding[~clusteredD, 0],
#             clusterable_embedding[~clusteredD, 1],
#             c='k',#(0.5, 0.5, 0.5),
#             s=0.1,
#             alpha=0.5)
# plt.scatter(clusterable_embedding[clusteredD, 0],
#             clusterable_embedding[clusteredD, 1],
#             c=labels[clusteredD],
#             s=0.1,
#             cmap='viridis');

# outfile = os.path.join(dirname, basename)+'.covd-clustering.png'
# plt.savefig(outfile) # save as png
# plt.close()

# ####################################################################################################
# # Color nodes by labels
# ###################################################################################################

# print('Color the graph by descriptor cluster')

# node_cluster_color = -1.0*np.ones(A.shape[0]) #check
# ind = 0                                     # this is the subgraph label
# for nodes in graph2covd:                    # for each subgraph and corresponding covariance matrix
#     node_cluster_color[nodes] = labels[ind] # update node_color with cluster labels for each node in the subgraph
#     ind += 1

# clusteredN = (node_cluster_color >= 0) # bolean array with true for clustered nodes and false for the rest
# print(str(sum(clusteredN))+' nodes clustered of '+str(clusteredN.shape[0])+' in total')

# non_clusteredN = (node_cluster_color < 0) # bolean array with true for clustered nodes and false for the rest
# print(str(sum(non_clusteredN))+' nodes NOT clustered of '+str(clusteredN.shape[0])+' in total')

# clustered_nodes = np.asarray(list(G.nodes))[clusteredN] 
# clustered_nodes_color = node_cluster_color[clusteredN] 
# subG = G.subgraph(clustered_nodes)

# sns.set(style='white', rc={'figure.figsize':(50,50)})
# nx.draw_networkx_nodes(subG, pos, alpha=0.5,node_color=clustered_nodes_color, node_size=1,cmap='viridis')

# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

# plt.axis('off')
# outfile = os.path.join(dirname, basename)+'.node-clustering.png'
# plt.savefig(outfile, dpi=100) # save as png
# # plt.savefig(outfile, dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
# plt.close()

# ####################################################################################################
# # Color nodes not clustered before
# ###################################################################################################
# print('Determine the connected components of the non clustered nodes')

# non_clustered_nodes = np.asarray(list(G.nodes))[non_clusteredN] 
# subGnot = G.subgraph(non_clustered_nodes)
# graphs = [g for g in list(nx.connected_component_subgraphs(subGnot)) if g.number_of_nodes()>=20]

# print('Determine covariance matrix of the non clustered connected components')
# covdata, graph2covd = covd_multifeature(features,G,graphs) 
# logvec = [linalg.logm(m).reshape((1,covdata[0].shape[0]*covdata[0].shape[1])) for m in covdata] #calculate the logm and vectorize
# X_nonclustered = np.vstack(logvec) #create the array of vectorized covd data

# print('Cluster the other connected components by finding the min distance to the clustered subgraphs')
# from scipy import spatial
# reference = X[clusteredD,:]
# tree = spatial.KDTree(reference)
# for row_ind in range(X_nonclustered.shape[0]):
#     index = tree.query(X_nonclustered[row_ind,:])[1]
#     for nodes in graph2covd:                    
#         node_cluster_color[nodes] = labels[index] 

# sns.set(style='white', rc={'figure.figsize':(50,50)})
# nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_cluster_color, node_size=1,cmap='viridis')

# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

# plt.axis('off')
# outfile = os.path.join(dirname, basename)+'.all-node-clustering.png'
# plt.savefig(outfile, dpi=100) # save as png
# # plt.savefig(outfile, dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
# plt.close()



