#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy import sparse
import sys
import umap
import warnings
import networkx as nx
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, zscore, poisson
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

morphology = sys.argv[1] #id...txt.gz
W = sparse.load_npz(sys.argv[2]) #id...graph.npz
degreefile = sys.argv[3] #id...degree.gz
ccfile = sys.argv[4]  #id...cc.gz
steps = int(sys.argv[5]) #number of steps of the random walker

morpho = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(7,9,12,13))   # area,perimeter,circularity,eccentricity,intensity
degree_vec = np.loadtxt(degreefile, delimiter=" ").reshape((morpho.shape[0],1))
cc_vec = np.loadtxt(ccfile, delimiter=" ").reshape((morpho.shape[0],1))

data = np.hstack((morpho,degree_vec,cc_vec)) # stack morphology and topology

##################
# Should I use a row stochastic matrix or not? In principle is not necessary
#SS = normalize(W, norm='l1', axis=1) #create the row-stochastic matrix 
################

history = data
for counter in range(steps):
    newdata = W.dot(data) # !!!use weighted graph adj not the stochastic matrix for the moment!!!
    history = np.dstack((history,newdata))
#    print(history.shape)

if history.shape[0] == W.shape[0] and history.shape[2] == steps+1:
    print("Saving history...")
    filename = str(morphology)+'.walkhistory'    
    np.save(filename,history)
else:
    print(history.shape)
    print("Oops!  History is not in good shape. Check data or code...")

