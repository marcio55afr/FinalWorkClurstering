# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:09:03 2020

@author: marci
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def getDatasets():
    
    list_df = []
    list_df.append( pd.read_csv('Data/OlivierFeatures.csv') )
    list_df.append( pd.read_csv('Data/BooneFeatures.csv') )
    list_df.append( pd.read_csv('Data/Linfoma.csv') )
    list_df.append( pd.read_csv('Data/Displasia.csv') )

    return list_df


def fillNullValues(data):
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    # transform the dataset
    columns_ = data.columns
    dataFilled = pd.DataFrame( imputer.fit_transform(data),
                                columns = columns_ )
    return dataFilled