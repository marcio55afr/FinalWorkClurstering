# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:51 2020

@author: marci
"""

import ReadData
from ClusteringAlgorithms import KMeansModel, MeanShiftModel, DBSCANModel,SpectralClusteringModel, AffinityPropagationModel

list_df = ReadData.getDatasets()


for df, param in list_df:
    print('\n\n\n\n\tDataset: ',param.name)
    kmeans = KMeansModel(df)
    kmeans.fit(param.k_clusters)
    kmeans.printInfo()
    
    meanShift = MeanShiftModel(df)
    meanShift.fit()
    meanShift.printInfo()
    
    # dbscan = DBSCANModel(df)
    # dbscan.fit(param.eps)
    # dbscan.printInfo()
    
    # spectral_clustering = SpectralClusteringModel(df)
    # spectral_clustering.fit(param.k_clusters)
    # spectral_clustering.printInfo()
    
    affinity_propagation = AffinityPropagationModel(df)
    affinity_propagation.fit(param.damping)
    affinity_propagation.printInfo()