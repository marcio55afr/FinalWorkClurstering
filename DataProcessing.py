# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:50:53 2020

@author: tiago
"""

import pandas as pd
import ReadData
from Config import Parameters, Seed
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
    datasets = []
    for data, param in list_data[1:2]:
        print(data)
        #Scaling the samples to have unit norm
        normalization = ['l1', 'l2']
        for norm in normalization:
            data_normalized = preprocessing.normalize(data,
                                                      norm = norm,
                                                      axis = 0,
                                                      copy = True)
            name = 'data_normalized_' + norm
            data_normalized = pd.DataFrame(data_normalized, columns = data.columns)
            parameters = Parameters(name, 
                                    'Data/'+name+'.csv', 
                                    param.k_clusters,
                                    param.eps,
                                    param.damping)
            datasets.append( (data_normalized, parameters) )
        
        
        #Mapping data to a defined distribution by quantile transforms
        distribution = ['uniform','normal']
        for dist in distribution:
            data_normalized = preprocessing.quantile_transform(data,
                                                               axis = 0,
                                                               output_distribution = dist,
                                                               random_state = Seed,
                                                               copy=True)
            name = 'data_distribution_' + dist
            data_normalized = pd.DataFrame(data_normalized, columns = data.columns)
            parameters = Parameters(name, 
                                    'Data/'+name+'.csv', 
                                    param.k_clusters,
                                    param.eps,
                                    param.damping)
            datasets.append( (data_normalized, parameters) )
        
        '''
        #Mapping data to a defined distribution by power transforms
        standardize = [True,False]
        for stand in standardize:
            data_normalized = preprocessing.power_transform(data,
                                                            method = 'yeo-johnson',
                                                            standardize = stand,
                                                            copy=True)
            name = 'data_distribution_' + dist
            data_normalized = pd.DataFrame(data_normalized, columns = data.columns)
            parameters = Parameters(name, 
                                    'Data/'+name+'.csv', 
                                    param.k_clusters,
                                    param.eps,
                                    param.damping)
            datasets.append( (data_normalized, parameters) )
        '''
    return datasets
        



















        
        