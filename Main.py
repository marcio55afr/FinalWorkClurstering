# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:51 2020

@author: marci
"""

import ReadData
from ClusteringAlgorithms import kMeansModel

list_df = ReadData.getDatasets()

kmeans1 = kMeansModel(list_df[0], 36)
kmeans1.stardartProcess(14)
kmeans1.printInfo()

kmeans2 = kMeansModel(list_df[1], 3)
kmeans2.stardartProcess(14)
kmeans2.printInfo()

