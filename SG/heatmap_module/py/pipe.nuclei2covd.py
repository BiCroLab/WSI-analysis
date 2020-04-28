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

import h5py
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def covd(mat):
    ims = coo_matrix(mat)
    imd = np.pad( mat.astype(float), (1,1), 'constant')
    [x,y,I] = [ims.row,ims.col,ims.data]  

    Ix = [] #first derivative in x
    Iy = [] #first derivative in y
    Ixx = [] #second der in x
    Iyy = [] #second der in y 
    Id = [] #magnitude of the first der 
    Idd = [] #magnitude of the second der
    
    for ind in range(len(I)):
        Ix.append( imd[x[ind]+1,y[ind]] - imd[x[ind]-1,y[ind]] )
        Ixx.append( imd[x[ind]+1,y[ind]] - 2*imd[x[ind],y[ind]] + imd[x[ind]-1,y[ind]] )
        Iy.append( imd[x[ind],y[ind]+1] - imd[x[ind],y[ind]-1] )
        Iyy.append( imd[x[ind],y[ind]+1] - 2*imd[x[ind],y[ind]] + imd[x[ind],y[ind]-1] )
        Id.append(np.linalg.norm([Ix,Iy]))
        Idd.append(np.linalg.norm([Ixx,Iyy]))
    descriptor = np.array( list(zip(list(x),list(y),list(I),Ix,Iy,Ixx,Iyy,Id,Idd)),dtype='int64' ).T     # descriptors
    C = np.cov(descriptor) #covariance of the descriptor
    iu1 = np.triu_indices(C.shape[1]) # the indices of the upper triangular part
    covd2vec = C[iu1]
    return covd2vec

'''
Set the input information
'''
h5_file = sys.argv[1]   #this file contains the segmented nuclei
datadir = os.path.dirname(os.path.realpath(h5_file))
basename = os.path.splitext(h5_file)[0]
dapi_file = basename+'.tif' # the dapi file has to be located in the same directory as the h5 file
npz_file = basename #this is the output file with spatial and morphological descriptors
method = 'covd' #choose between covd rotational invariant or not: covdRI or covd 

fov = h5py.File(h5_file, 'r') # load the current fov segmentation
mask = fov['/exported_watershed_masks'][:]
mask_reduced = np.squeeze(mask, axis=2) #to get rid of the third dimension
dapi_fov= cv2.imread(dapi_file,cv2.IMREAD_GRAYSCALE) #the dapi tif file of the current FOV

#Check which position the current FOV occupies within the big scan
row = h5_file.split('_r',1)[1].split('_c')[0]
col = h5_file.split('_r',1)[1].split('_c')[1].split('.')[0]

# label all connected components in the fov, 0 is background
mask_label, numb_of_nuclei = label(mask_reduced,return_num=True) 

centroids = []    #list of centroid coordinates for sc in each fov
descriptors = []  #list of descriptors for sc in each fov
morphology = [] #list of morphology features for sc in each fov
counter=0
print('r:',row,'c:',col,'nuclei:',numb_of_nuclei)
for region in regionprops(mask_label,intensity_image=dapi_fov):
    counter+=1
    if ((np.count_nonzero(region.intensity_image) <= 10) or (np.count_nonzero(region.intensity_image) > 2500)) :        
        print('The number of pixels is '+str(np.count_nonzero(region.intensity_image))+' in region='+str(counter))
    else:
        x = 512*int(col)+region.centroid[0] # shift by FOV location
        y = 512*int(row)+region.centroid[1] # shift by FOV location
        centroids.append((x,y))

        morphology.append((region.area,region.perimeter,region.solidity,region.eccentricity,region.mean_intensity))
        
        descriptors.append(covd(region.intensity_image))

#save covd to file
from datetime import datetime
dateTimeObj = datetime.now() # Returns a datetime object containing the local date and time

if numb_of_nuclei > 0:
    np.savez(str(npz_file)+'_'+str(method)+'.npz',
             centroids=centroids,
             descriptors=descriptors,
             morphology=morphology)
else:
    print('There are no nuclei in row='+str(row)+' and col='+str(col)+' in file: '+str(h5_file))

with open(basename+'.txt', 'a+', newline='') as myfile:
     wr = csv.writer(myfile)
     wr.writerow([dateTimeObj,'row='+str(row),'col='+str(col),'nuclei='+str(numb_of_nuclei),'#descriptors='+str(len(descriptors))])
     

