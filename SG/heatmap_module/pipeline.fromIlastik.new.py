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

basename_graph = os.path.splitext(os.path.basename(filename))[0]
basename_smooth = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)
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
# Features list = fov_name  x_centroid      y_centroid      area    perimeter       eccentricity    circularity     mean_intensity  total_intensity
from sklearn.preprocessing import normalize 
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(3,4,5,6,7,8)).reshape((A.shape[0],6))
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

####################################################################################################
# Reweight the graph
###################################################################################################
print('Rescale graph weights by local morphology')
ww = []
morphology = normalize(morphology, norm='l1', axis=0) # normalize features
morphology_smooth = normalize(morphology_smooth, norm='l1', axis=0) # normalize features
GG = copy.deepcopy(G)
for ijw in G.edges(data='weight'):
    # !!!identical morphologies are discarded at the moment!!!
    feature = np.asarray([ abs(morphology[ijw[0],f]-morphology[ijw[1],f]) for f in range(morphology.shape[1]) ]) # array of morphology features 

    GG[ijw[0]][ijw[1]]['weight'] = ijw[2]/np.sum(feature) # the new graph weights

####################################################################################################
# Community detection
###################################################################################################
print('Find the communities')
from cdlib import algorithms
import igraph

CC = list(nx.connected_component_subgraphs(GG))

outfile = os.path.join(dirname, basename)+'.communities'
if os.path.exists(outfile+'.npy'):
    communities = np.load(outfile+'.npy',allow_pickle=True)
else:
    communities = []
    for graph in CC:
        file_edgelist = str(outfile)+'.edge_list.csv'
        nx.write_edgelist(graph,file_edgelist,data=['weight']) # writhe the edge list 
        g = igraph.Graph.Read_Ncol(file_edgelist, directed = False, weights = True)
        weights = g.es['weight']
        coms = algorithms.leiden(g,weights='weight')
        communities.append(coms.communities)
    np.save(outfile, communities)

bigcommunities = [sg for g in communities for sg in g if len(sg) > threshold] # flatten list of communities
outfile = os.path.join(dirname, basename)+'.bigcommunities'
np.save(outfile, bigcommunities)
print('There are '+str(len(bigcommunities))+' big communities and '+str(len([sg for g in communities for sg in g]))+' communities in total')

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
