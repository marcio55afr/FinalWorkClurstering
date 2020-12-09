# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:51 2020

@author: marci
"""

import ReadData
import numpy as np
import pandas as pd
from ClusteringAlgorithms import KMeansModel, MeanShiftModel, DBSCANModel,SpectralClusteringModel, AffinityPropagationModel
from sklearn import preprocessing

list_df = ReadData.getDatasets()

def searchBestEps():
    datasetMetrics = {}
    for df, param in ReadData.getDatasets()[:2]:
        print("searching eps for dataset: ", param.name)
        metrics = []
        for eps_i in np.arange(0.1,10,0.1):
            if((eps_i % 1) == 0):
                print(str(eps_i * 10),'%')
            dbscan = DBSCANModel(df)
            dbscan.fit(eps_i)
            metrics.append(dbscan.getValidation())
        datasetMetrics[param.name] = pd.DataFrame(metrics,columns = 
                                                  ['pureza','entropia','rand index',
                                                   'silhueta','davies-bouldin'],
                                                  index = np.arange(0.1,10,0.1))
    return datasetMetrics
        

def main():
    for df, param in list_df:
        print('\n\n\n\n\tDataset: ',param.name)
        # kmeans = KMeansModel(df)
        # kmeans.fit(param.k_clusters)
        # kmeans.printInfo()
        
        # meanShift = MeanShiftModel(df)
        # meanShift.fit()
        # meanShift.printInfo()
        
        dbscan = DBSCANModel(df)
        dbscan.fit(10)
        dbscan.printInfo()
        
        # spectral_clustering = SpectralClusteringModel(df)
        # spectral_clustering.fit(param.k_clusters)
        # spectral_clustering.printInfo()
        
        # affinity_propagation = AffinityPropagationModel(df)
        # affinity_propagation.fit(param.damping)
        # affinity_propagation.printInfo()
        
metrics = searchBestEps()
# main()
