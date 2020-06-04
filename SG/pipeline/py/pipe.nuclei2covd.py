#!/usr/bin/env python
# coding: utf-8

import numpy as np
from skimage.measure import label, regionprops
from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
from scipy.linalg import eigh, inv, logm, norm
from  scipy import ndimage,sparse
import cv2
import os
import sys
import csv
import glob
import math
import h5py
from matplotlib import pyplot as plt
from graviti import *
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

'''
Set the input information
'''
h5_file = sys.argv[1]   # this file contains the segmented nuclei

datadir = os.path.dirname(os.path.realpath(h5_file)) # the directory of the h5 file
basename = os.path.splitext(h5_file)[0]              # the main name of the h5 file
dapi_file = basename+'.tif' # the dapi file has to be located in the same directory as the h5 file
npz_file = basename         # this is the output file with spatial and morphological descriptors
#method = 'covd' #choose between covd rotational invariant or not: covdRI or covd 

fov = h5py.File(h5_file, 'r') # load the current fov segmentation
mask = fov['/exported_watershed_masks'][:]
mask_reduced = np.squeeze(mask, axis=2)              # to get rid of the third dimension
dapi_fov= cv2.imread(dapi_file,cv2.IMREAD_GRAYSCALE) # the dapi tif file of the current FOV

#Check which position the current FOV occupies within the big scan
row = h5_file.split('_r',1)[1].split('_c')[0]
col = h5_file.split('_r',1)[1].split('_c')[1].split('.')[0]

# label all connected components in the fov, 0 is background
mask_label, numb_of_nuclei = label(mask_reduced,return_num=True) 

fov = []          # list of fov locations
centroids = []    # list of centroid coordinates for sc in each fov
descriptors = []  # list of descriptors for sc in each fov
counter=0

print('r:',row,'c:',col,'nuclei:',numb_of_nuclei)

for region in regionprops(mask_label,intensity_image=dapi_fov):
    counter+=1
    if not ((np.count_nonzero(region.intensity_image) <= 10) or (np.count_nonzero(region.intensity_image) > 2500)) :        
        fov.append((int(row),int(col)))
        
        x = 512*int(col)+region.centroid[0] # shift by FOV location
        y = 512*int(row)+region.centroid[1] # shift by FOV location
        centroids.append((x,y))
        descriptors.append(covd(region.intensity_image))
        
dateTimeObj = datetime.now() # Returns a datetime object containing the local date and time

# Save information to file
if numb_of_nuclei > 0:
    np.savez(str(npz_file)+'.intensity-covd.npz',
             fov=fov,
             centroids=centroids,
             descriptors=descriptors)
# else:
#     print('There are no nuclei in row='+str(row)+' and col='+str(col)+' in file: '+str(h5_file))

     

