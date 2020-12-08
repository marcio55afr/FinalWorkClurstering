# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:51 2020

@author: marci
"""

import ReadData
from ClusteringAlgorithms import KMeansModel, MeanShiftModel, OpticsModel

list_df = ReadData.getDatasets()

kmeans_objects = []
meanShift_objects = []
optics_objects = []
opticsDbscan_objects = []

for df, param in list_df:
    print('\n\n\n\n\tDataset: ',param.name)
    kmeans = KMeansModel(df)
    kmeans.fit(param.k_clusters)
    kmeans.printInfo()
    kmeans_objects.append(kmeans)
    
    meanShift = MeanShiftModel(df)
    meanShift.fit()
    meanShift.printInfo()
    meanShift_objects.append(meanShift)
    
    optics = OpticsModel(df)
    optics.fit()
    optics.printInfo()
    optics_objects.append(optics)
    
    
    # Didn' work, dbscan is clurstering all data as one cluster
    # and some information cannot calculate that.
    
    #optics = OpticsModel(df)
    #optics.fit_dbscan()
    #optics.printInfo()
    #opticsDbscan_objects.append(optics)