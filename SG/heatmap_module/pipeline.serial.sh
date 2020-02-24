#!/usr/bin/env bash

id=$1     # .txt.gz file
stain=$2  # HE or DAPI
steps=$3  # radius of the subgraph for averaging features
if [ -d /usr/local/share/anaconda3 ]; then
    path2anaconda=/usr/local/share/anaconda3/bin
    echo The python executable directory is $path2anaconda
fi
if [ -d /home/garner1/miniconda3 ]; then
    path2anaconda=/home/garner1/miniconda3/bin
    echo The python executable directory is $path2anaconda
fi

path2data=/media/garner1/hdd2/$stain/segmentation
#path2data=.
echo The python data path is $path2data

if [ -f ${id} ]; then
    if [ ! -f ${id}.nn10.adj.npz ]; then
	echo Make Graph
	$path2anaconda/python3.7 pipe3.makeGraph.fromQuPath.py ${id}
    fi
    echo Make average windows
    $path2anaconda/python3.7 pipe4.walk.fromQuPath.py ${id} ${id}{.nn10.adj.npz,.nn10.degree.gz,.nn10.cc.gz} $steps
    # echo Make heatmaps
    # parallel "$path2anaconda/python3.7 pipe5.drawHeatMap.fromQuPath.py $path2data/${id}__*{.npy,.txt.gz} 5 {} ${id} deciles linear noflip ./" ::: 1 10 100 1000
else
    echo $path2data/${id}__*.txt.gz does not exist!
fi
