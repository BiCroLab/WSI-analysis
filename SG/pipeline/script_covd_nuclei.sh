#!/usr/bin/env bash

# for id in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Download data for sample ${id}
#     basedir='/media/server2_projects/CNA/Digital_pathology/LungBrain_cancer/Adenocarcinoma'
#     mkdir id_${id}
#     cp ${basedir}/Patient_${id}/Watershed_${id}/iMS*._r*_c*.h5 id_${id}
#     cp ${basedir}/Patient_${id}/Filtered_${id}/iMS*._r*_c*.tif id_${id}
# done

# echo Calculate measurements features CovD
# parallel "/usr/local/share/anaconda3/bin/python3.7 py/pipe.nuclei2features.py {}" ::: data/id_*/iMS*._r*_c*.h5

# for id in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     /usr/local/share/anaconda3/bin/python3.7 py/pipe.laplacianField.py ${id} 0 10 50
# done

# echo Calculate intensity arrays Covd
# parallel "/usr/local/share/anaconda3/bin/python3.7 py/pipe.nuclei2covd.py {}" ::: data/id_*/iMS*._r*_c*.h5
# parallel "/usr/local/share/anaconda3/bin/python3.7 py/pipe.laplacianField.intensity.py {} 0 10 50" ::: 52 57 38 53 40 17 39 13 54 41 51 56 45 46

###########################################################
# for sample in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Run UMAP on sample ${sample}
#     /usr/local/share/anaconda3/bin/python3.7 py/pipe.features2data.py data/id_${sample} ${sample}
# done

# for sample in 52 57 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Run UMAP on sample ${sample}
#     time /usr/local/share/anaconda3/bin/python3.7 py/pipe.data2cluster.py pkl/id_${sample}.measurements.covd.pkl 1000 200 1000 200
# done

# for sample in 52 57 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Run UMAP on sample ${sample}
#     /usr/local/share/anaconda3/bin/python3.7 py/pipe.cluster2mask.py pkl/id_${sample}.measurements.covd.pkl.intensityANDmorphology.csv.gz data_intensity/id_${sample}
# done

# parallel "/usr/local/share/anaconda3/bin/python3.7 py/intensity.cluster2mask.py id_{}.fov_centroids_embedding_morphology.covd.pkl.intensity.csv.gz data_intensity/id_{}" ::: 52 57 38 40 17 39 13 54 41 51 56 45 46
