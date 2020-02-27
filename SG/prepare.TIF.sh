#!/usr/bin/env bash

file=$1 #input file

convert ${file} -define tiff:tile-geometry=128x128 -depth 8 ptif:${file}.8bit.convert.tif #does not support nd2
~/tools/bioimageconvert/imgcnv -i ${file}.8bit.convert.tif -o ${file}.8bit.convert.imgcnv.tif -t TIFF -options compression none tiles 512 pyramid subdirs

