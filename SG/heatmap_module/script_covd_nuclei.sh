#!/usr/bin/env bash

for id in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
do
    echo Download data for sample ${id}
    basedir='/media/server2_projects/CNA/Digital_pathology/Lung&Brain_cancer/Adenocarcinoma'
    mkdir id_${id}
    cp ${basedir}/Patient_${id}/Watershed_${id}/iMS*._r*_c*.h5 id_${id}
    cp ${basedir}/Patient_${id}/Filtered_${id}/iMS*._r*_c*.tif id_${id}
done

echo Calculate covd
parallel "/usr/local/share/anaconda3/bin/python3.7 pipe.nuclei2covd.py {}" ::: id_*/iMS*._r*_c*.h5
echo Run UMAP
parallel "/usr/local/share/anaconda3/bin/python3.7 pipe.covd2data.py id_{} {}" ::: 52 57 38 53 40 17 39 13 54 41 51 56 45 46
