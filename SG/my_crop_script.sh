#!/usr/bin/env bash

file=$1
height=$(/usr/bin/tiffinfo $file | grep Length | awk '{print $NF}')
width=$(/usr/bin/tiffinfo $file | grep Width | awk '{print $3}')

xlist=$(seq 0 10000 $width)
xlast=$(seq 0 10000 $width | tail -1)

for x0 in $xlist; do
    echo $x0
    if [[ $x0 == 0 ]]
    then
	xstart=$x0
	lwidth=10000
	batch_crop $xstart 0 $lwidth $height $file
    fi
    if [[ $x0 > 0 ]] && [[ $x0 < $xlast ]]
    then
	xstart=$(($x0 - 1000))
	lwidth=11000	
	batch_crop $xstart 0 $lwidth $height $file
    fi
    if [[ $x0 == $xlast ]]
    then
	xstart=$(($x0 - 1000))
	lwidth=$(($width - $xstart))
	batch_crop $xstart 0 $lwidth $height $file
    fi
done
