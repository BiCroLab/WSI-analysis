from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import sys
import umap
import warnings
from scipy import sparse
warnings.filterwarnings('ignore')

# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10) 
def processInput(data,vec,col,ind):
    M = data[ind]
    v = vec[col[ind]]
    return np.multiply(M,v).shape
 
W = sparse.load_npz(sys.argv[1]).tocoo() #id...graph.npz
vecfile = sys.argv[2] #id...degree.gz
vec = np.loadtxt(vecfile, delimiter=" ").reshape((W.shape[0],1))

data = W.data
row = W.row
col = W.col

num_cores = multiprocessing.cpu_count()
chuncks = np.array_split(range(len(row)), num_cores)
  
results = Parallel(n_jobs=num_cores)(delayed(processInput)(data,vec,col,ind) for ind in chuncks)
print(results)


