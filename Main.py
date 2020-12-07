# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:51 2020

@author: marci
"""

import ReadData
from ClusteringAlgorithms import kMeansModel

list_df = ReadData.getDatasets()

kmeans_objects = []

for df, param in list_df:
    kmeans = kMeansModel(df)
    kmeans.stardartProcess(param.k_clusters)
    kmeans.printInfo(param.name)
    
    kmeans_objects.append(kmeans)

