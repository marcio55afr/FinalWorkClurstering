# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:50:53 2020

@author: tiago
"""

import pandas as pd
import ReadData
import Config
from sklearn import preprocessing

def rearrange_Columns():
    
    return 1    
    #data = pd.read_csv('Data/___.csv',header = 0,index_col = 0)
    # columns = list(data.columns)
    # columns = columns[0:-3] + columns[-2:]
    # data = data[columns]
    # columns[-1] = 'target'
    # data.columns = columns
    
    #data.to_csv('Data/___.csv',index = False)

def createNormalizedDatasets():
    
    list_data = ReadData.getDatasets()
    
    for data, param in list_data:
        
        #Scaling the samples to have unit norm
        normalization = ['l1', 'l2', 'max']
        axis = [0,1]
        for ax in axis:
            for norm in normalization:
                data_normalized = preprocessing.normalize(data,
                                                          norm = norm,
                                                          axis = ax,
                                                          copy = True)
                path = 'Data/' + 'data_normalized_' + norm + 'axis' + str(ax) +'.csv'
                data_normalized = pd.DataFrame(data_normalized, columns = data.columns)
                data_normalized.to_csv(path, index = False)
        
        
        #Mapping data to a defined distribution by quantile transforms
        distribution = ['uniform','normal']
        for ax in axis:
            for dist in distribution:
                data_normalized = preprocessing.quantile_transform(data,
                                                                   axis = ax,
                                                                   output_distribution = dist,
                                                                   random_state = Config.Seed,
                                                                   copy=True)
                path = 'Data/' + 'data_distribution_' + dist + 'axis' + str(ax) +'.csv'
                data_normalized = pd.DataFrame(data_normalized, columns = data.columns)
                data_normalized.to_csv(path, index = False)
        
        
        #Mapping data to a defined distribution by power transforms
        standardize = [True,False]
        for stand in standardize:
            data_normalized = preprocessing.power_transform(data,
                                                            method = 'yeo-johnson',
                                                            standardize = stand,
                                                            copy=True)
            path = 'Data/' + 'data_distribution_' + dist + 'axis1.csv'
            data_normalized = pd.DataFrame(data_normalized, columns = data.columns)
            data_normalized.to_csv(path, index = False)
        
createNormalizedDatasets()   



















        
        