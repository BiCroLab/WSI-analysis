## Software used
* vips
* exiftool
* bioimage converter
* UMAP
* HDBSCAN

## Pipeline
### Segmentation
.svs files and .nd2 files need to be processed differently. 
* svs files can be all processed in qupath 
* small nd2 files (< 2GB) can be processed in qupath directly
* large nd2 files (> 2GB) are converted into tif, split into vertical chuncks and then processed in qupath 
### Morphology table
* Morphology table generated by qupath 
* Morphology table generated by skimage.measure on ilastik segmentation
### Processing
* Covariance descriptor of the intensity array of each nucleous
* UMAP reduced representation of the descriptors
* Density based hierarchical clustering of the low-curvature sector
* WSI and FOV annotations
