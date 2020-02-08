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

filename = sys.argv[1] #MN35B__953995-35B.svs.Detections.txt
mat_XY = sparse.load_npz(sys.argv[2]) #MN35B__953995-35B.svs.Detections.txt_graph.npz
feature = sys.argv[3] #Area Perimeter Circularity Eccentricity Intensity
steps = sys.argv[4] #number of steps of the random walker 

XY = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(5,6))
SS = normalize(mat_XY, norm='l1', axis=1) #create the row-stochastic matrix

if feature == 'area':
    vec = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(7))
elif feature == 'perimeter':
    vec = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(8))
elif feature == 'circularity':
    vec = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(9))
elif feature == 'eccentricity':
    vec = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(12))
elif feature == 'intensity':
    vec = np.loadtxt(sys.argv[1], delimiter="\t",skiprows=True,usecols=(13))

history = vec
nn = int(steps)
for counter in range(nn):
    vec = SS.dot(vec)
    history = np.hstack((history,vec)) 

filename = str(sys.argv[1])+'.'+str(feature)+'.walkhistory'    
np.save(str(filename),history)
