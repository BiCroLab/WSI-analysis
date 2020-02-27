#!/usr/bin/env bash

nd2=$1

xmax=$(~/tools/bioimageconvert/imgcnv -i $nd2 -meta | grep image_num_x | cut -d' ' -f2)
ymax=$(~/tools/bioimageconvert/imgcnv -i $nd2 -meta | grep image_num_y | cut -d' ' -f2)

~/tools/bioimageconvert/imgcnv -i $nd2 -o $nd2.out -tile 15000 
