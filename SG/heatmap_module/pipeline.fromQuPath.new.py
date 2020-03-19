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
print('Topological graph ready!')
print('The graph has '+str(A.shape[0])+' nodes')
pos2norm = np.linalg.norm(pos,axis=1).reshape((pos.shape[0],1)) # the modulus of the position vector

####################################################################################################
# Select the morphological features,
# and set the min number of nodes per subgraph
###################################################################################################

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
    morphology_smooth = np.load(outfile+'.npy')
else:
    morphology_smooth = smoothing(A, morphology, radius)
    np.save(outfile, morphology_smooth)

####################################################################################################
# Reweight the graph
###################################################################################################
ww = []
morphology = normalize(morphology, norm='l1', axis=0)
morphology_smooth = normalize(morphology_smooth, norm='l1', axis=0)
for ijw in G.edges(data='weight'):
    # feature = np.asarray([ abs(morphology_smooth[ijw[0],f]-morphology_smooth[ijw[1],f]) for f in range(morphology_smooth.shape[1]) ]) # array of morphology features 
    feature = np.asarray([ abs(morphology[ijw[0],f]-morphology[ijw[1],f]) for f in range(morphology.shape[1]) if morphology[ijw[0],f] != morphology[ijw[1],f]]) # array of morphology features 

    G[ijw[0]][ijw[1]]['weight'] = ijw[2]/np.prod(feature) # the new graph weights

####################################################################################################
# Community detection
###################################################################################################

from cdlib import algorithms
import igraph

CC = list(nx.connected_component_subgraphs(G))

outfile = os.path.join(dirname, basename)+'.communities'
if os.path.exists(outfile+'.npy'):
    commmunities = np.load(outfile+'.npy')
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


