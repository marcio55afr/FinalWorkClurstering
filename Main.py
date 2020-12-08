# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:51 2020

@author: marci
"""

import ReadData
from ClusteringAlgorithms import KMeansModel, MeanShiftModel, DBSCANModel

list_df = ReadData.getDatasets()

kmeans_objects = []
meanShift_objects = []
dbscan_objects = []

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
    
    # dbscan = DBSCANModel(df)
    # dbscan.fit()
    # dbscan.printInfo()
    # dbscan_objects.append(dbscan)