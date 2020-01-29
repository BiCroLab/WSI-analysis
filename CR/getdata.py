#!/usr/bin/env python
# coding: utf-8

#Extract properties from watershed
import h5py
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix
import cv2
import os
import sys

#Sort .h5 and .tif datasets
def custom_sorth5(name_fov): 
    [other,value]=name_fov.split('sub')
    [value,other,other]=value.split('_')    
    return value
def custom_sortdapi(name_dapi): 
    [other,value]=name_dapi.split('sub')
    [value,other]=value.split('.')    
    return value


datadir = '/home/garner1/Work/dataset/tissue2graph'
h5_file = sys.argv[1]
dapi_file = sys.argv[2]

#Get all the FOVs and arrange them in the position
#To get the range we have to add one
h5_files = []
for file in os.listdir(datadir):
    if file.endswith(".h5"):
        h5_files.append(os.path.join(datadir, file))
number_fovs=np.arange(1,len(h5_files)+1) #these are the labels of the fov
fov_matrix=np.array(number_fovs).reshape(23,25) #reshape according to the patches 2D structure

sintensity=[] #Create a list in which to save the sparses
row_list = [] #list of rows coordinates of the fov
col_list = [] #list of cols coordinates of the fov
centroids = []

# for current_fov in h5_files[:2]:
fov = h5py.File(h5_file, 'r')

mask=fov['/exported_watershed_masks'][:]
mask_reduced=np.squeeze(mask, axis=2)

dapi_fov= cv2.imread(dapi_file,cv2.IMREAD_GRAYSCALE) #Get DAPI fov 

#Check which position the FOV occupies within the big scan
#Position of FOV ilastik_masks_watershed1.h5
[other,value]=dapi_file.split('sub')
[value,other]=value.split('.')
(row,col)=np.where(fov_matrix==int(value))
mask_label=label(mask_reduced) # label all connected components in the fov, 0 is background

centroids = [] #list of centroid coordinates for sc in each fov
for region in regionprops(mask_label):
        centroid = region.centroid
        centroids.append((int(centroid[0]),int(centroid[1])))

# type(mask)
# [height,width]=mask_reduced.shape

#Create a 3D sparse array where x,y are FOV size and z is the amount of nuclei in the FOV 
sintensity=[]
z_dimension=0 #Set a counter to check current stack
for i in range(1,np.amax(mask_label)+1): # 0 is background so it doesn't get included in the range
    xmask,ymask=np.where(mask_label==i)
    single_cell_mask=lil_matrix((1024, 1024), dtype='uint8')
    single_cell_mask[xmask,ymask]=dapi_fov[xmask,ymask]
    sintensity.append(single_cell_mask) #Add current nuclei sparse on to the FOV array
    z_dimension+=1 #Move to the next stack (next nuclei label)

#Save properties
np.savez(dapi_file+'_data.npz',sintensity=sintensity,row=row,col=col,centroids=centroids)


