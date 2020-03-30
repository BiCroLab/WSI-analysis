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
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(1,2)) # chose x and y and check if header is present or not
    G = nx.read_gpickle(os.path.join(dirname, basename_graph) + ".graph.pickle")
    d = getdegree(G)
    cc = clusteringCoeff(A)
else:
    print('The graph does not exists yet')
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(1,2)) # here there is no header
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

pos2norm = np.linalg.norm(pos,axis=1).reshape((pos.shape[0],1)) # the modulus of the position vector

print('Topological graph ready!')
print('The graph has '+str(A.shape[0])+' nodes')

####################################################################################################
# Select the morphological features,
# and set the min number of nodes per subgraph
###################################################################################################
print('Prepare the morphology array')
# Features list = fov_name  x_centroid      y_centroid      area    perimeter       eccentricity    circularity   mean_intensity  total_intensity
from sklearn.preprocessing import normalize 
# morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(3,4,5,6,7,8)).reshape((A.shape[0],6))
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(3,5,6,8)).reshape((A.shape[0],4))
threshold = max(radius,(morphology.shape[1]+4)*2) # set the min subgraph size based on the dim of the feature matrix

####################################################################################################
# Reweight the graph
###################################################################################################
print('Rescale graph weights by local morphology')

print('...use the raw morphology...')
morphology_normed = normalize(morphology, norm='l1', axis=0) # normalize features
GG = copy.deepcopy(G)
for ijw in G.edges(data='weight'):
    feature = np.asarray([ abs(morphology_normed[ijw[0],f]-morphology_normed[ijw[1],f]) for f in range(morphology_normed.shape[1]) ]) # array of morphology features
    GG[ijw[0]][ijw[1]]['weight'] = ijw[2]/(1.0+np.sum(feature)) # the new graph weights is the ratio between the current and the sum of the neightbours differences

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

print('...generate connected components as subgraphs...')
graphs = list(nx.connected_component_subgraphs(GG))

print('...convert networkx graph to igraph object...')
communities = []
for graph in graphs:
    nx.write_weighted_edgelist(graph, basename+".edgelist.txt")
    g = igraph.Graph.Read_Ncol(basename+".edgelist.txt", names=True, weights="if_present", directed=False)
    os.remove(basename+".edgelist.txt")
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,initial_membership=None, weights='weight', seed=rndsample, n_iterations=2)
    communities.extend([g.vs[x]['name'] for x in part])
    print( 'The number of communities is '+str(len(communities)) )

bigcommunities = [g for g in communities if len(g) > threshold] # list of big enough communities
outfile = os.path.join(dirname, basename)+'.bigcommunities'
np.save(outfile, bigcommunities)

print('There are '+str(len(bigcommunities))+' big communities and '+str(len(communities))+' communities in total')

####################################################################################################
# Generate the covariance descriptors
# this can be done with respect to raw features or smoothed ones
###################################################################################################
print('Generate the covariance descriptor')
#!!! Should positions be included?  !!!
# features = np.hstack((pos2norm,morphology))            # this is rotational invariant
features = morphology            # this is rotational invariant

outfile_covd = os.path.join(dirname, basename)+'.covd.npy'
if os.path.exists(outfile_covd):
    print('... loading the descriptors ...')
    covdata = np.load(outfile_covd,allow_pickle=True)
else:
    print('... creating the descriptors ...')
    # !!! you can use G or GG here  !!!
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
    logvec = [ linalg.logm(m).reshape((1,covdata[0].shape[0]*covdata[0].shape[1]))  for m in covdata] #calculate the logm and vectorize
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

# ####################################################################################################
# # Free up spaces
# ###################################################################################################
# del G                           # G is not needed anymore
# del A                           # A is not needed anymore
# del morphology

# ####################################################################################################
# # Color graph nodes by community label
# ###################################################################################################
# print('Preparing to color the graph communities')
# print('...set up the empty graph...')
# g = nx.Graph()
# g.add_nodes_from(range(pos.shape[0])) # add all the nodes of the graph, but not all of them are in a covd cluster because of small communities
# print('...set up the empty dictionary...')
# dictionary = {}
# for node in range(pos.shape[0]):
#     dictionary[int(node)] = -1 # set all node to -1

# print('...set up the full dictionary...')
# node_comm_tuples = [(int(node),i) for i, community in enumerate(bigcommunities) for node in community]
# dictionary.update(dict(node_comm_tuples))

# node_color = []
# for i in sorted (dictionary) :  # determine the color based on the community
#     node_color.append(dictionary[i])

# print('...draw the graph...')
# sns.set(style='white', rc={'figure.figsize':(50,50)})
# nx.draw_networkx_nodes(g, pos, alpha=0.5,node_color=node_color, node_size=1,cmap=plt.cm.Set1)

# print('...saving graph...')
# plt.axis('off')
# plt.savefig(os.path.join(dirname, basename)+'.community_graph.png') # save as png
# plt.close()
