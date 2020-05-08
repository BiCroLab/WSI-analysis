#!/usr/bin/env bash

parallel -j 6 "/usr/local/share/anaconda3/bin/python3.7 pipeline.fromIlastik.py ~/Work/dataset/ilastik/id57/nuclei57.txt.woInf.gz 1 {}" ::: `seq 1 24`
parallel -j 6 "/usr/local/share/anaconda3/bin/python3.7 pipeline.fromIlastik.py ~/Work/dataset/ilastik/id52/nuclei52.txt.woInf.gz 1 {}" ::: `seq 1 24`
parallel -j 4 "/usr/local/share/anaconda3/bin/python3.7 pipeline.fromQuPath.py /media/garner1/hdd2/HE/detections/MN52__18732_15-52.svs.Detections.txt.gz 1 {}" ::: `seq 1 24`
parallel -j 4 "/usr/local/share/anaconda3/bin/python3.7 pipeline.fromQuPath.py /media/garner1/hdd2/HE/detections/MN57__949233-57.svs.Detections.txt.gz 1 {}" ::: `seq 1 24`

# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.py /media/garner1/hdd2/HE/detections/MN{}__*.svs.Detections.txt.gz 100 10" ::: 52 57
# /usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.py /media/garner1/hdd2/HE/detections/MN2__*.svs.Detections.txt.gz 100 10
# /usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.py /media/garner1/hdd2/HE/detections/MN4__*.svs.Detections.txt.gz 100 10

# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN{}__*.svs.Detections.txt.gz 50 10 50" ::: 2 4 

# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN{}__*.svs.Detections.txt.gz 50 10 50" ::: 52 57 


# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN52__*.svs.Detections.txt.gz {} 10 20" ::: 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN52__*.svs.Detections.txt.gz {} 10 50" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN52__*.svs.Detections.txt.gz {} 10 100" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN52__*.svs.Detections.txt.gz {} 10 500" ::: 10 100

# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN2__*.svs.Detections.txt.gz {} 10 20" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN2__*.svs.Detections.txt.gz {} 10 50" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN2__*.svs.Detections.txt.gz {} 10 100" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN2__*.svs.Detections.txt.gz {} 10 500" ::: 10 100

# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN4__*.svs.Detections.txt.gz {} 10 20" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN4__*.svs.Detections.txt.gz {} 10 50" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN4__*.svs.Detections.txt.gz {} 10 100" ::: 10 100
# parallel "/usr/local/share/anaconda3/bin/python3.7 pipeline.woPCA.bis.py /media/garner1/hdd2/HE/detections/MN4__*.svs.Detections.txt.gz {} 10 500" ::: 10 100

