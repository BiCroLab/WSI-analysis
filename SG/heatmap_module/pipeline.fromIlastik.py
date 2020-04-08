#!/usr/bin/env python

####################################################################################################
# Load the necessary libraries
###################################################################################################

import networkx as nx
import numpy as np
from scipy import sparse, linalg
from sklearn.preprocessing import normalize 

from graviti import *           # the local module

import sys, getopt 
import os
import copy
import seaborn as sns; sns.set()

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

####################################################################################################
# Define the input parameters
####################################################################################################

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:s:n:a:p:e:c:m:t:",
                               ["input","seed=","nn=","area=","perimeter=",
                                "eccentricity=","circularity=","meanIntensity=","totalIntensity=","pos"])
except getopt.GetoptError as err:
    # print help information and exit:
    print(str(err))  # will print something like "option -a not recognized"
    usage()
    sys.exit(2)
usecols = ()
for o, a in opts:
    if o in ("-i", "--input"):
        filename = a            # morphology measurements file
    elif o in ("-s", "--seed"):
        rndsample = int(a)      # Leiden seed 
    elif o in ("-n", "--nn"):
        nn = int(a)      # Leiden seed 
    elif o in ("-a", "--area"):
        if a == '1': usecols = usecols+(3,)
    elif o in ("-p", "--perimeter"):
        if a == '1': usecols = usecols+(4,)
    elif o in ("-e", "--eccentricity"):
        if a == '1': usecols = usecols+(5,)
    elif o in ("-c", "--circularity"):
        if a == '1': usecols = usecols+(6,)
    elif o in ("-m", "--meanIntensity"):
        if a == '1': usecols = usecols+(7,)
    elif o in ("-t", "--totalIntensity"):
        if a == '1': usecols = usecols+(8,)
    elif o in ("--pos"):
        if a == '1':
            position = True
        else:
            position = False
    else:
        assert False, "unhandled option"

####################################################################################################
# Define basic filenames
# !!! have the base name dependent on the parameters !!!
####################################################################################################

basename_graph = os.path.splitext(os.path.basename(filename))[0]
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.s'+str(rndsample)
dirname = os.path.dirname(filename)

####################################################################################################
# Construct the UMAP graph
# and save the adjacency matrix 
# and the degree and clustering coefficient vectors
###################################################################################################
print('Prepare the topological graph ...')

path = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.adj.npz'

if os.path.exists(path) and os.path.exists( os.path.join(dirname, basename_graph) + ".graph.pickle" ):
    print('The graph exists already and I am now loading it...')
    A = sparse.load_npz(path) 
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(1,2)) # chose x and y and do not consider header
    G = nx.read_gpickle(os.path.join(dirname, basename_graph) + ".graph.pickle")
    d = getdegree(G)
    cc = clusteringCoeff(A)
else:
    print('The graph does not exists yet and I am going to create one...')
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(1,2)) 
    A = space2graph(pos,nn)     # create the topology graph
    sparse.save_npz(path, A)
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    d = getdegree(G)
    cc = clusteringCoeff(A)
    outfile = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.degree.gz'
    np.savetxt(outfile, d)
    outfile = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.cc.gz'
    np.savetxt(outfile, cc)
    nx.write_gpickle(G, os.path.join(dirname, basename_graph) + ".graph.pickle")

pos2norm = np.linalg.norm(pos,axis=1).reshape((pos.shape[0],1)) # the length of the position vector

print('Topological graph ready!')
print('...the graph has '+str(A.shape[0])+' nodes')

####################################################################################################
# Select the morphological features,
# and set the min number of nodes per subgraph
# Features list:
# fov_name  x_centroid      y_centroid      area    perimeter       eccentricity    circularity   mean_intensity  total_intensity
# !!!optimize the threshold!!!
###################################################################################################
print('Prepare the morphology array')
print('...the features tuple that we consider is: ',str(usecols))
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=usecols).reshape((A.shape[0],len(usecols)))

threshold = (morphology.shape[1]+4)*2 # set the min subgraph size based on the dim of the feature matrix
morphologies = morphology.shape[1]    # number of morphological features

####################################################################################################
# Weight the graph taking into account topology and morphology
# the new weight is the ratio topological_similarity/(1+morpho_dissimilarity)
# !!!need to be optimized!!!
###################################################################################################
print('Rescale graph weights by local morphology')

morphology_normed = normalize(morphology, norm='l1', axis=0) # normalize features
GG = copy.deepcopy(G)                                        # create a topology+morphology new graph

for ijw in G.edges(data='weight'): # loop over edges
    feature = np.asarray([ abs(morphology_normed[ijw[0],f]-morphology_normed[ijw[1],f]) for f in range(morphologies) ]) 
    GG[ijw[0]][ijw[1]]['weight'] = ijw[2]/(1.0+np.sum(feature)) 

####################################################################################################
# Community detection in the topology+morphology graph
# !!! find a way to avoid writing the edge list on disk !!!
###################################################################################################
print('Find the communities in GG')

from cdlib import algorithms
from cdlib import evaluation
from cdlib.utils import convert_graph_formats
import igraph
import leidenalg
from networkx.algorithms.community.quality import modularity

print('...generate connected components as subgraphs...')
graphs = list(nx.connected_component_subgraphs(GG)) # list the connected components

print('...convert networkx graph to igraph object...')
communities = []
for graph in graphs:
    nx.write_weighted_edgelist(graph, basename+".edgelist.txt") # write the edge list on disk
    g = igraph.Graph.Read_Ncol(basename+".edgelist.txt", names=True, weights="if_present", directed=False) # define the igraph obj
    os.remove(basename+".edgelist.txt") # delete the edge list
    part = leidenalg.find_partition(g,
                                    leidenalg.ModularityVertexPartition,
                                    initial_membership=None,
                                    weights='weight',
                                    seed=rndsample,
                                    n_iterations=2) # find partitions
    communities.extend([g.vs[x]['name'] for x in part]) # create a list of communities

bigcommunities = [g for g in communities if len(g) > threshold] # list of big enough communities

outfile = os.path.join(dirname, basename)+'.bigcommunities'
np.save(outfile, bigcommunities) # store the big communities

print('There are '+str(len(bigcommunities))+' big communities and '+str(len(communities))+' communities in total')

####################################################################################################
# Generate the covariance descriptors of the topology graph
# !!! insert a switch for the position !!!
###################################################################################################
print('Generate the covariance descriptor')

if position:
    print('...the position information is included')
    features = np.hstack((pos2norm,morphology))            # this is rotational invariant
else:
    print('...the position information is not included')
    features = morphology            # this is without positions

outfile_covd = os.path.join(dirname, basename)+'.covd.npy' # name of the covd file
if os.path.exists(outfile_covd):
    print('... loading the descriptors ...')
    covdata = np.load(outfile_covd,allow_pickle=True) # load covd data
else:
    print('... creating the descriptors ...')
    covdata = community_covd(features,G,bigcommunities) # get list of cov matrices and a list of nodes per matrix
    np.save(outfile_covd,covdata)                       # store covd data

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
    logvec = [ linalg.logm(m).reshape((1,m.shape[0]*m.shape[1]))  for m in covdata] #calculate the logm and vectorize
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
