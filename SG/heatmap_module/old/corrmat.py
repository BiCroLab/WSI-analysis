#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd
############################################
data = np.load(sys.argv[1]) # localdata.npy
print(data.shape)
if len(data.shape) == 3:
    for time in range(3):
        timedata = data[:,:,time]
        df = pd.DataFrame.from_records(timedata)
        df.columns = ['area','perimeter','circularity','eccentricity','mean_intensity','degree','cc']
        print(df.corr(method='pearson'))
if len(data.shape) == 2:
    df = pd.DataFrame.from_records(data)
    df.columns = ['area','perimeter','circularity','eccentricity','mean_intensity','degree','cc']
    print(df.corr(method='pearson'))

