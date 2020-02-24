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

npyfilename = sys.argv[1] # 'walkhistory.npy'
txtfilename = sys.argv[2] #'txt.gz'
feature = int(sys.argv[3]) # 0 area,1 perimeter,2 circularity,3 eccentricity,4 intensity,5 degree,6 cc
steps = int(sys.argv[4]) # 0: 5, 1: 50, 2:500
modality = sys.argv[5] #absolute or deciles division of the feature range
scale = sys.argv[6] #linear of logarithmic scale of the attribute values

history = np.load(npyfilename,allow_pickle=True)

if scale == 'linear':
    attribute = history[:,feature,steps]
    print(attribute.shape)
elif scale == 'logarithmic':
    attribute = np.log2(history[:,feature,steps])
    attribute = attribute[np.isfinite(attribute)]

# create empty list for node colors
G = nx.Graph()
pos = np.loadtxt(txtfilename, delimiter="\t",skiprows=True,usecols=(5,6))
G.add_nodes_from(range(len(attribute)))

# color attribute based on percentiles, deciles or quartiles ...
if modality == 'absolute':
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
plt.savefig(txtfilename+".heatmap-"+str(feature)+".scale-"+str(scale)+".partition-"+str(modality)+".radius-"+str(steps)+".png", dpi=100,bbox_inches = 'tight', pad_inches = 0.5) # save as png
plt.close()

#########################################
##########################################                       
# # Fit a normal distribution to the data:
# mu, std = norm.fit(attribute) # you could also fit to a lognorma the original data
# sns.set(style='white', rc={'figure.figsize':(5,5)})
# plt.hist(attribute, bins=100, density=True, alpha=0.6, color='g')
# #Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)
# plt.savefig(outdir+'/'+str(ID)+"_distro-"+str(scale)+"_scale.png") # save as png
# plt.close()
###########################################

