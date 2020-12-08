# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:50:53 2020

@author: tiago
"""

import pandas as pd


data = pd.read_csv('Data/OlivierFeatures.csv',header = 0,index_col = 0)
# columns = list(data.columns)
# columns = columns[0:-3] + columns[-2:]
# data = data[columns]
# columns[-1] = 'target'
# data.columns = columns

data.to_csv('Data/OlivierFeatures.csv',index = False)