# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:09:03 2020

@author: marci
"""

import pandas as pd
import numpy as np
import Config
from sklearn.impute import SimpleImputer

def getDatasets():
    
    list_df = []
    for config_dataset in Config.datasets:
        list_df.append( (pd.read_csv(config_dataset.path), config_dataset) )

    return list_df


def fillNullValues(data):
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    # transform the dataset
    columns_ = data.columns
    dataFilled = pd.DataFrame( imputer.fit_transform(data),
                                columns = columns_ )
    return dataFilled