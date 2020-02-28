#!/usr/bin/env bash

file=$1
######################
# Correct resolution in the output cropped images:
xresolution=$(exiftool -s $file | grep XResolution | awk '{print $NF}') #get the correct values
yresolution=$(exiftool -s $file | grep YResolution | awk '{print $NF}') #get the correct values
resolutionunit=$(exiftool -s $file | grep ResolutionUnit | awk '{print $NF}') #get the correct values

filename=$(basename $file)
dirname=$(dirname $file)

for f in $dirname/cropped/crop_*_$filename;
do
    exiftool -exif:YResolution=$xresolution $f #substitute the correct value
    exiftool -exif:XResolution=$yresolution $f #substitute the correct value
    exiftool -exif:resolutionunit=$resolutionunit $f #substitute the correct value
    rm ${f}_original
done
