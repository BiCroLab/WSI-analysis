#!/usr/bin/env bash

tiffdir=$1 #dir containing pyramid tiff files

mother=/usr/local/share/anaconda3/bin/  #path/to/anaconda/bins
deep=/home/garner1/openslide-python/examples/deepzoom #path/to/deepzoom

${mother}/python3 ${deep}/deepzoom_multiserver.py -Q 100 ${tiffdir}
