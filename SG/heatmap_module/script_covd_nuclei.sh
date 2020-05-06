#!/usr/bin/env bash

# for id in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Download data for sample ${id}
#     basedir='/media/server2_projects/CNA/Digital_pathology/Lung&Brain_cancer/Adenocarcinoma'
#     mkdir id_${id}
#     cp ${basedir}/Patient_${id}/Watershed_${id}/iMS*._r*_c*.h5 id_${id}
#     cp ${basedir}/Patient_${id}/Filtered_${id}/iMS*._r*_c*.tif id_${id}
# done

# echo Calculate covd
# parallel "/usr/local/share/anaconda3/bin/python3.7 py/intensity.nuclei2covd.py {}" ::: data_intensity/id_*/iMS*._r*_c*.h5

# echo Run UMAP
# parallel "/usr/local/share/anaconda3/bin/python3.7 py/intensity.covd2data.py data_intensity/id_{} {}" ::: 52 57 38 53 40 17 39 13 54 41 51 56 45 46

#for sample in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# for sample in 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Run UMAP on sample ${sample}
#     /usr/local/share/anaconda3/bin/python3.7 py/intensity.covd2data.py data_intensity/id_${sample} ${sample}
# done

# for sample in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Run UMAP on sample ${sample}
#     /usr/local/share/anaconda3/bin/python3.7 py/intensity.data2cluster.py id_${sample}.fov_centroids_embedding_morphology.covd.pkl
# done

# for sample in 52 57 38 53 40 17 39 13 54 41 51 56 45 46;
# do
#     echo Run UMAP on sample ${sample}
#     /usr/local/share/anaconda3/bin/python3.7 py/intensity.cluster2mask.py id_${sample}.fov_centroids_embedding_morphology.covd.pkl.intensity.csv.gz data_intensity/id_${sample}
# done
parallel "/usr/local/share/anaconda3/bin/python3.7 py/intensity.cluster2mask.py id_{}.fov_centroids_embedding_morphology.covd.pkl.intensity.csv.gz data_intensity/id_{}" ::: 52 57 38 40 17 39 13 54 41 51 56 45 46
