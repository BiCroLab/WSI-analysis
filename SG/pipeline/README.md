# Unsupervised clustering of nuclei in WSI 

This pipeline receives as input segmented nuclei from WSI, and it uses their intensities and morphological features to cluster them in different groups.

## I pipe: nuclei2covd
Map the nuclei intensity arrays into a descriptor's space.

* Input: segmented nuclei (.h5 files)

* Output: descriptor

## II pipe: covd2data
Evaluate the reduced representation of the descriptor, 
and format together morphologies and the reduced representation.

* Input: morphopologies and descriptors

* Output: a pkl table

## III pipe: data2cluster

Density based hierarchical clustering of the UMAP reduced representation

* Input: the UMAP reduced representation

* Output: a csv table with cluster ID

## IV pipe: cluster2mask
Visualize clusters and different nuclei masks in single FOVs 

* Input: clusters

* Output: .png files with dapi and colored mask, .tiff files with different clusters in different channels

