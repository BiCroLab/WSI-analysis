# Morphological heterogeneity of nuclei in WSI 

This pipeline receives as input segmented nuclei from WSI, and it uses their intensities and morphometric features to define an heterogeneity metric at the desired length-scale. The logarithmic euclidean distance is used to evaluate the metric, and since its computation scale with the cube of the size of the nucleus descriptor, we have parallelized the pipeline in the relevant computational steps.   

## Subsampling
To speed up computation we recommend to subsample the total set of N segmented nuclei into a smaller S set of nuclei. Typically 10^5 nuclei from a total of few milions are enough to provide an informative representation of the morphological heterogeneity of a WSI. 

## Construction of the morphological graph
Given the centroid coordinates of the S sampled nuclei the topological graph is constructed following the first steps of the UMAP representation. The same [motivations behind the UMAP graph construction](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) apply in our case (namely a robust mathematical construction to deal with noisy data). The graph has S nodes, and a number of edges per node that can be specified by the user (typically 10 is enough). Each edge is weighted by a real number between 0 and 1 that specifies the proximity of the two nodes.     

## Covariance descriptor
To each node i of the graph is associated a morphological descriptor C<sub>i</sub>, which is the covariance matrix of the normalized measurements of the node i and its k nearest neighbors. These k nearest neighbors are taken from the set of all N segmented nuclei. The parameter k is set in order to provide a good coverage of the entire WSI. The covariance descriptor is a sort of morphological smoothing filter that provides information about the coherence structure of different morphological features for a local set of segmented nuclei, moreover the metric structure of the space of covariance matrices can be used to properly quantify similarity between (possibly overlapping) regions around different nuclei.  

## Edge and node heterogeneity metric
Given the morphological descriptor of each node/nucleus in the graph, we can evaluate the morphological dissimilarity between any pairs of nodes, taking into account all the features that have been measured. 
The space of covariance descriptors can be endowed with different metrics, each with their specific advantages and disadvantages (both in terms of invariance with respect to specific operations and computational efficiency). 
We use the Log-Euclidean distance of the covariance descriptors 
<img src="https://render.githubusercontent.com/render/math?math=h^{edge}_{ij}=\| Log(C_i)-Log(C_j) \|_2">
to quantify morphological heterogeneity between the connected nodes i and j. We weight this measure of purely morphological hetergeneity with the spatial information provided by w<sub>ij</sub>, the edge weight on the topological graph (providing a normalized measure of local spatial proximity).

The heterogeneity at node i is defined as the sum over all incident edges heterogeneities:
<img src="https://render.githubusercontent.com/render/math?math=h^{node}_{i}=\sum_j \| Log(C_i)-Log(C_j) \|_2">. 
It is an isotropic measure of morphological heterogeneity at a given spatial location. 
