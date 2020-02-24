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
# feature = sys.argv[6] #one of area,perimeter,circularity,eccentricity,intensity,degree,cc

# if feature == 'area':
#     data = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(7,))#.reshape((W.shape[0],1))
# if feature == 'perimeter':
#     data = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(8,))#.reshape((W.shape[0],1))
# if feature == 'circularity':
#     data = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(9,))#.reshape((W.shape[0],1))
# if feature == 'eccentricity':
#     data = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(12,))#.reshape((W.shape[0],1))
# if feature == 'intensity':
#     data = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(13,))#.reshape((W.shape[0],1))
# if feature == 'degree':
#     data = np.loadtxt(degreefile, delimiter=" ")#.reshape((W.shape[0],1))
# if feature == 'cc':
#     data = np.loadtxt(ccfile, delimiter=" ")#.reshape((W.shape[0],1))

morpho = np.loadtxt(morphology, delimiter="\t", skiprows=True, usecols=(7,8,9,12,13)).reshape((W.shape[0],5))
degree_vec = np.loadtxt(degreefile, delimiter=" ").reshape((W.shape[0],1))
cc_vec = np.loadtxt(ccfile, delimiter=" ").reshape((W.shape[0],1))
data = np.hstack((morpho,degree_vec,cc_vec))
print(data.shape)
##################
# Should I use a row stochastic matrix or not? In principle is not necessary
#SS = normalize(W, norm='l1', axis=1) #create the row-stochastic matrix 
################

# history = np.zeros((data.shape[0],data.shape[1],steps+1))
# history[:,:,0] = data
# for counter in range(steps):
#     print(morphology+str( counter))
#     history[:,:,counter+1] = W.dot(history[:,:,counter])

#Save only 3 point averages otherwise you need 75G per patient
smooth = np.zeros((data.shape[0],data.shape[1],3))
for counter in range(steps):
    newdata = W.dot(data)
    data = data+newdata
    if counter == 4:
        smooth[:,:,0] = data
    if counter == 49:
        smooth[:,:,1] = data
    if counter == 499:
        smooth[:,:,2] = data

print("Saving history...")
filename = str(morphology)+'.walkhistory'    
np.save(filename,smooth)

