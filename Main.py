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
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

list_df = ReadData.getDatasets()


def searchEps():
    for df, param in ReadData.getDatasets():
        df = (df - df.mean())/df.std() 
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(df)
        distances, indices = neighbors_fit.kneighbors(df)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.figure()
        plt.xlabel('amostras')
        plt.ylabel('k-ésimo vizinho')
        plt.title(param.name)
        plt.plot(distances)
        plt.show()
        
def searchBestDumping():
    datasetMetrics = {}
    linfoma = ReadData.getDatasets()[:1]
    for df, param in linfoma:
        print("searching Dumping for dataset: ", param.name)
        metrics = []
        dumps = []
        for dump in np.arange(0.5,1,0.01):
            if(dump>=1): dump=0.99
            if((dump % 0.1) == 0):
                print(str(dump * 20),'%')
            affinity = AffinityPropagationModel(df)
            affinity.fit(dump)
            metrics.append(affinity.getValidation())
        datasetMetrics[param.name] = pd.DataFrame(metrics,columns = 
                                                  ['pureza','entropia','rand index',
                                                   'silhueta','davies-bouldin','n_groups'],
                                                  index = np.arange(0.5,1,0.01))
        
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
        
# metrics = searchEps()


main()








