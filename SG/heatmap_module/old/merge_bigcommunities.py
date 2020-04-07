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
if os.path.splitext(os.path.basename(filename))[1] == '.gz':
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+'.r'+str(radius)
dirname = os.path.dirname(filename)

####################################################################################################
# Construct the UMAP graph
# and save the adjacency matrix 
# and the degree and clustering coefficient vectors
###################################################################################################
print('Prepare the topological graph ...')
nn = 10 # this is hardcoded at the moment
path = os.path.join(dirname, basename_graph)+'.nn'+str(nn)+'.adj.npz'

print('Loading the graph exists already')
A = sparse.load_npz(path) #id...graph.npz
pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(1,2)) # chose x and y and check if header is present or not
G = nx.read_gpickle(os.path.join(dirname, basename_graph) + ".graph.pickle")
d = getdegree(G)
cc = clusteringCoeff(A)
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
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(3,4,5,6,7,8)).reshape((A.shape[0],6))
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
print('Find the communities in G')

from cdlib import algorithms
from cdlib import evaluation
from cdlib.utils import convert_graph_formats
import igraph
import leidenalg
from networkx.algorithms.community.quality import modularity
import glob

print('Loading random communities')
bigcommunities = []
for c in glob.glob(os.path.join(dirname, os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0])+r'*.bigcommunities.npy'):
    rnd_comm = np.load(c,allow_pickle=True)
    bigcommunities.extend(rnd_comm)
    print(len(bigcommunities))

####################################################################################################
# Generate the covariance descriptors
# this can be done with respect to raw features or smoothed ones
###################################################################################################
print('Generate the covariance descriptor')
features = np.hstack((pos2norm,morphology))            # this is rotational invariant

print('... creating the descriptors ...')
# !!! you can use G or GG here  !!!
covdata = community_covd(features,G,bigcommunities) # get list of cov matrices and a list of nodes per matrix

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
print('...create the logvec dataset...')
logvec = [linalg.logm(m).reshape((1,covdata[0].shape[0]*covdata[0].shape[1]))  for m in covdata] #calculate the logm and vectorize
X = np.vstack(logvec) #create the array of vectorized covd data

print('The vectorized covd array has shape '+str(X.shape))    
outfile_clusterable_embedding = os.path.join(dirname, basename)+'.clusterable_embedding.merge-rnd.npy'

print('...create the clusterable embedding...')
clusterable_embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(X) # this is used to identify clusters

np.save( outfile_clusterable_embedding,clusterable_embedding )

print('The embedding has shape '+str(clusterable_embedding.shape))

