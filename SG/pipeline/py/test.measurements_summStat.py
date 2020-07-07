#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import glob

sample_idx = 0
for filename in glob.glob('/media/garner1/hdd2/tcga.detection/*.gz'):
    if sample_idx == 0:
        df = pd.read_csv(filename,sep='\t')
        features = df.columns[np.r_[0, 7:18]]
        df_pool = df[features].describe().reset_index()
        df_pool['Image'] = [df[df.columns[0]].iloc[0]]*df_pool.shape[0]
    else:
        df = pd.read_csv(filename,sep='\t')
        df_summary = df[features].describe().reset_index()
        df_summary['Image'] = [df[df.columns[0]].iloc[0]]*df_summary.shape[0]
        df_pool = pd.concat([df_pool, df_summary], ignore_index=True)
    print(sample_idx,df_pool.shape)
    sample_idx += 1

df_pool.to_pickle("/media/garner1/hdd2/pooled_measurements_min-25-50-75-max.pkl")
