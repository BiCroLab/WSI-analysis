#!/usr/bin/env bash

file=$1 # the name of the original non-cropped file

# Correct resolution in the output cropped images:
xresolution=$(exiftool -s $file | grep XResolution | awk '{print $NF}') #get the correct values

filename=$(basename $file)
dirname=$(dirname $file)

echo $xresolution $filename $dirname

for f in $dirname/crop_*_$filename.Detections.txt;
do
    pixel_offset=$(echo $f | cut -d'_' -f2 | cut -d'-' -f1 | sed 's/L//')
    if [[ ${pixel_offset} -eq 0 ]] # the first crop is not changed
    then
	# shift the cropped files by the offset calculated above and list them together in a single file
	cat $f | tr ' ' '_' > $dirname/$filename.Detections.txt
    fi
    if [[ ${pixel_offset} -gt 0 ]] #only the subsequent crops are changed
    then
	offset=`echo "scale=2 ; ${pixel_offset} / ${xresolution}" | bc`
	# shift the cropped files by the offset calculated above and list them together in a single file
	tail -n+2 $f | tr ' ' '_' | awk -v offset=${offset} '{$6=$6+offset; print $0}' | tr ' ' '\t' >> $dirname/$filename.Detections.txt
    fi
done
