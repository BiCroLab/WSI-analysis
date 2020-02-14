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

npyfilename = sys.argv[1] #'/home/garner1/Work/pipelines/graviti/heatmap_module/npy/ID57-area-walkhistory.npy'
txtfilename = sys.argv[2] #'/home/garner1/Work/dataset/tissue2graph/ID57_data.txt'
scale = sys.argv[3]
modality = sys.argv[4]

cc = np.load(npyfilename,allow_pickle=True)
print(cc.shape)
if scale == 'linear':
    attribute = cc
elif scale == 'logarithmic':
    attribute = np.log2(cc)
    attribute = attribute[np.isfinite(attribute)]
    
##########################################                       
# Fit a normal distribution to the data:
mu, std = norm.fit(attribute) # you could also fit to a lognorma the original data
sns.set(style='white', rc={'figure.figsize':(5,5)})
plt.hist(attribute, bins=100, density=True, alpha=0.6, color='g')
#Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.savefig("myCC_distro-"+str(scale)+"_scale"+".png") # save as png
plt.close()
###########################################

# create empty list for node colors
G = nx.Graph()
pos = np.loadtxt(txtfilename, delimiter="\t",skiprows=True,usecols=(5,6))

G.add_nodes_from(range(len(attribute)))

# color attribute based on percentiles, deciles or quartiles ...
if modality == 'linear':
    node_color = np.interp(attribute, (attribute.min(), attribute.max()), (0, +10))
elif modality == 'deciles':
    node_color = pd.qcut(attribute, 10, labels=False)

# draw graph with node attribute color
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=1,cmap='viridis')

print('saving graph')

plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.axis('off')
plt.savefig("myCC__"+str(scale)+"_scale-"+str(modality)+"_partition.png", dpi=200,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()

