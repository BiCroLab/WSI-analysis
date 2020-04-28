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
                                "eccentricity=","circularity=","meanIntensity=","totalIntensity=","pos="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(str(err))  # will print something like "option -a not recognized"
    usage()
    sys.exit(2)
print(opts)
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
    basename = os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]
dirname = os.path.dirname(filename)

####################################################################################################
# Cluster the morphology table
###################################################################################################

print('Clustering the morphology')
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.cluster import OPTICS

print('...the features tuple that we consider is: ',str(usecols))
morphology = np.loadtxt(filename, delimiter="\t", skiprows=True, usecols=usecols)
print(morphology.shape)
morphology_normed = normalize(morphology, norm='l1', axis=0) # normalize features
print(morphology_normed.shape)


print('...create the clusterable embedding...')
outfile_clusterable_embedding = os.path.join(dirname, 'clusterable_embedding.morphology.npy')

clusterable_embedding = umap.UMAP(min_dist=0.0,n_components=3,random_state=42).fit_transform(morphology_normed) # this is used to identify clusters

np.save( outfile_clusterable_embedding, clusterable_embedding )

print('The embedding has shape '+str(clusterable_embedding.shape))
