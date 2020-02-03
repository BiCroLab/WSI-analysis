#!/usr/bin/env bash

tif=$1 #origanal tiff

~/tools/bioimageconvert/imgcnv -i $tif -o $tif.2.tif -t TIFF -options compression lzw tiles 128 pyramid subdirs

~/tools/bioimageconvert/imgcnv -i $tif.2.tif -o $tif.2.jpeg -t JPEG
