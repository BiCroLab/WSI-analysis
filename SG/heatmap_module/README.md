# Heatmap visualization of the whole slide graph-representation 

This pipeline makes use of segmented nuclei and their morphological attributes in order to visualize the WSI as a graph whose nodes are the nuclei and whose edges are the UMAP encoded distances between nuclei. Each node is color coded based on the value of its feature (area, intensity, perimeter, eccentricity, solidity).

## I pipe
```
$ path/to/anaconda/bin/python3.7 pipe1.img2features.py /path/to/h5_dir/segmented_#row-#col_tile.h5 /path/to/tif_dir/#row-#col_tile.tif /path/to/npz_dir/#row-#col_tile patID_report_file
```
* Input: /path/to/h5_dir/segmented_#row-#col_tile.h5 /path/to/tif_dir/#row-#col_tile.tif
* Output: /path/to/npz_dir/#row-#col_tile patID_report_file

## Input

* list of h5 files with segmented nuclei
* list of tif images matching the h5 files

## Parameters

* (input) directory containing h5 files
* (input) directory containing tif files
* (output) directory containing npz files
* id label of the WSI
* file name of the report file

## How to run the code
```
$ bash pipeline ~/Work/dataset/tissue2graph/tissues/ID2/{h5,tif,npz} ID2
```

## The pipeline
* pipe \#1: collect centroids,areas and intensities for all the nuclei in the WSI
* pipe \#2: stitches the features collected in the previous step to annotate the WSI at the single nucleus scale
* pipe \#3: construct the UMAP graph from the centroids of the nuclei


