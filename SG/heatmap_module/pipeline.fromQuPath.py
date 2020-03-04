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
radius = int(sys.argv[2])   # for smoothing
quantiles = int(sys.argv[3]) # for stratifing the projection

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
print('Prepare the topological graph ...')
nn = 10 # this is hardcoded at the moment
path = os.path.join(dirname, basename)+'.nn'+str(nn)+'.adj.npz'
if not os.path.exists(path):
    print('The graph does not exists yet')
    A, pos = space2graph(filename,nn)
    sparse.save_npz(path, A)
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    d = getdegree(G)
    cc = clusteringCoeff(A)
    outfile = os.path.join(dirname, basename)+'.nn'+str(nn)+'.degree.gz'
    np.savetxt(outfile, d)
    outfile = os.path.join(dirname, basename)+'.nn'+str(nn)+'.cc.gz'
    np.savetxt(outfile, cc)
    nx.write_gpickle(G, os.path.join(dirname, basename) + ".graph.pickle")
if os.path.exists(path):
    print('The graph exists already')
    A = sparse.load_npz(path) #id...graph.npz
    pos = np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6))
    if os.path.exists( os.path.join(dirname, basename) + ".graph.pickle" ):
        print('A networkx obj G exists already')
        G = nx.read_gpickle(os.path.join(dirname, basename) + ".graph.pickle")
    else:
        print('A networkx obj G is being created')
        G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
        nx.write_gpickle(G, os.path.join(dirname, basename) + ".graph.pickle")
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

outfile = os.path.join(dirname, basename)+'.r'+str(radius)+'.smooth'
if os.path.exists(outfile+'.npy'):
    morphology_smooth = np.load(outfile+'.npy')
else:
    morphology_smooth = smoothing(A, morphology, radius)
    np.save(outfile, morphology_smooth)

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
# print(pca.explained_variance_ratio_)
# print(pca.n_components_,pca.n_features_,pca.n_samples_)
# print(pca.singular_values_)
# print(pca.components_)

####################################################################################################
# Project principal components back to real space
###################################################################################################
print('Project back to real space')
import pandas as pd

projection = np.dot(morphology_scaled,pca.components_.transpose()[:,0]) #project only the first PC
node_color = pd.qcut(projection, quantiles, labels=False)
print('Done!')

####################################################################################################
# Partition the graph
###################################################################################################
print('Generate the covariance descriptor')
outfile = os.path.join(dirname, basename)+'.covd.npy'
threshold = 100
if os.path.exists(outfile):
    covdata = np.load(outfile,allow_pickle=True)
else:
    features = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=(5,6,7,8,9,12,13,14)).reshape((A.shape[0],8)) #including X,Y
    covdata = covd(features,G,threshold,quantiles,node_color)
    np.save(outfile,covdata)

print('Done!')



    # print('Saving graph')
    # sns.set(style='white', rc={'figure.figsize':(50,50)})
    # nx.draw_networkx_nodes(subG, pos, alpha=0.5,node_color='r', node_size=1)
    
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # plt.axis('off')
    # outfile = os.path.join(dirname, basename)+'.heatmap'
    # plt.savefig("subgraph.q"+str(q)+".png", dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
    # plt.close()
    # print('Done!')
####################################################################################################
# Draw graph with node attribute color
###################################################################################################
# print('Saving graph')
# sns.set(style='white', rc={'figure.figsize':(50,50)})
# nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=1,cmap='viridis')

# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

# plt.axis('off')
# outfile = os.path.join(dirname, basename)+'.heatmap'
# plt.savefig(outfile+".q"+str(quantiles)+".r"+str(radius)+".png", dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
# plt.close()
# print('Done!')
###################################################################################################
###################################################################################################

# projection = np.dot(morphology_scaled,pca.components_.transpose()) #project only the first PC
# booleanProj = (projection > 0)
# uniqueValues, occurCount = np.unique(booleanProj, axis=0, return_counts=True)
# listOfUniqueValues = zip(uniqueValues, range(uniqueValues.shape[0]))
# print('Unique Values along with their cluster id')
# positions = [] # a list of list of indexes
# node_color = np.zeros((booleanProj.shape[0],1))
# for elem in listOfUniqueValues:
#    newlist = np.where(np.all(booleanProj==elem[0],axis=1))[0].tolist() #row index where a cluster occur
#    node_color[newlist] = elem[1]
#    positions.append(newlist)
# node_color = node_color.flatten() # to be used in colormap 
