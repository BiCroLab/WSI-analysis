# Heatmap visualization of the whole slide graph-representation 

This pipeline makes use of segmented nuclei and their morphological attributes in order to visualize the WSI as a graph whose nodes are the nuclei and whose edges are the UMAP encoded distances between nuclei. Each node is color coded based on the value of its feature (area, intensity, perimeter, eccentricity, solidity).

## I pipe
```
$ path/to/anaconda/bin/python3.7 pipe1.img2features.py /path/to/h5_dir/segmented_#row-#col_tile.h5 /path/to/tif_dir/#row-#col_tile.tif /path/to/npz_dir/#row-#col_tile patID_report_file
```
Input: 
* /path/to/h5_dir/segmented_#row-#col_tile.h5: is the segmented tile with the row-col location on the WSI specified in the filename
* /path/to/tif_dir/#row-#col_tile.tif: is the tif file matching the segmented tile with the row-col location on the WSI specified in the filename

Output: 
* /path/to/npz_dir/#row-#col_tile: the output npz file with morphological information for the given tile
* patID_report_file: output report with number of nuclei per tile

## II pipe
```
$ path/to/anaconda/bin/python3.7 pipe2.stitching.py /path/to/npz_dir 512 {id}
```
Input: 
* /path/to/npz_dir: path to the directory containing the .npz files with morphological information
* 512: linear size of the tiles
* id: the id uniquely associated to the WSI

Output:
* /path/to/id_data.npz: file containing the morphological information of all the nuclei in the WSI

## III pipe
```
$ path/to/anaconda/bin/python3.7 pipe3.makeGraph.py /path/to/id_data.npz {id}
```
Input:
* /path/to/id_data.npz: file containing the morphological information of all the nuclei in the WSI
* {id}: the id uniquely associated to the WSI

Output:
* /path/to/id_graph.npz: file containing the sparse adjacency matrix of the WSI

## IV pipe
```
$ path/to/anaconda/bin/python3.7 pipe4.walk.py /path/to/id_data.npz /path/to/id_graph.npz {feature} {steps} {id}
```
Input:
* /path/to/id_data.npz: file containing the morphological information of all the nuclei in the WSI
* /path/to/id_graph.npz: file containing the sparse adjacency matrix of the WSI
* {feature}: one of (area, intensity, perimeter, eccentricity, solidity)
* {steps}: extension of the random walk on the graph (ie 1000)
* {id}: the id uniquely associated to the WSI

Output:
* path/to/npy/{id}-{feature}-walkhistory.npy: file containing the random walk information  

## V pipe
```
$ path/to/anaconda/bin/python3.7 pipe5.drawHeatMap.py path/to/npy/{id}-{feature}-walkhistory.npy /path/to/id_data.npz {feature} {step} {id} {heatmap_scale} {feature_scale}
```
Input:
* path/to/npy/{id}-{feature}-walkhistory.npy: file containing the random walk information  
* /path/to/id_data.npz: file containing the morphology information 
* {feature}: one of (area, intensity, perimeter, eccentricity, solidity)
* {step}: extension of the random walk at which to plot the heatmap (it must be smaller than {steps} in pipe IV) 
* {id}: the id uniquely associated to the WSI 
* {heatmap_scale}: one of (linear, percentiles) 
* {feature_scale}: one of (linear, logarithmic)

Output:
* path/to/png/{id}_distro-{feature}-{feature_scale}-nn{step}.png
* path/to/png/{id}_heatmap-{feature}-{feature_scale}-{heatmap_scale}-nn{step}.png

