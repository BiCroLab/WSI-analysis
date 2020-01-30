#!/usr/bin/env bash

img=$1  #iMS337_20190709_001.tif
pyramid=$2  #iMS337.pyramid.tif

mother=/usr/local/share/anaconda3/bin/  #path/to/anaconda/bins
deep=/home/garner1/openslide-python/examples/deepzoom #path/to/deepzoom

[[ -d ${mother} ]] || echo ${mother} does not exists!
[[ -d ${deep} ]] || echo ${deep} does not exists!
[[ -f ${img} ]] || echo ${img} does not exists!
[[ -f ${pyramid} ]] || vips tiffsave ${img} ${pyramid} --tile --tile-width=256 --tile-height=256 --pyramid

