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
import time

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
    linfoma = ReadData.getDatasets()
    
    for df, param in linfoma:
        print("searching Damping for dataset: ", param.name)
        metrics = []
        path = 'Experiments/SearchingDamping/Affinity_bin/'+param.name        
        affinity_matrix = pd.read_csv(path, header=None)
        for damp in np.array(range(50,96,1))/100:
            if(damp>=1): damp=0.99
            if((damp % 0.1) <= 0.01):
                print(str(int(damp * 200 - 100)),'%')
            if(affinity_matrix is None):    
                affinity = AffinityPropagationModel(df)
                affinity.fit(damp, param.affinity['preference'])
                affinity_matrix = pd.DataFrame(affinity.model.affinity_matrix_)
                affinity_matrix.to_csv(path,index=False,header=False)
            else:                             
                affinity = AffinityPropagationModel(df)
                affinity.fit(damp, param.affinity['preference'], 'precomputed', affinity_matrix)
               
            metrics.append(affinity.getValidation())
        datasetMetrics[param.name] = (pd.DataFrame(metrics,columns = 
                                                  ['pureza','entropia','rand index',
                                                   'silhueta','davies-bouldin','n_groups'],
                                                  index = np.array(range(50,96,1))/100),
                                      param)
        
    return datasetMetrics

def main():
    for df, param in list_df:
        print('\n\n\n\n\tDataset: ',param.name)
        kmeans = KMeansModel(df)
        kmeans.fit(param.k_clusters)
        kmeans.printInfo()
        
        dbscan = DBSCANModel(df)
        dbscan.fit(10)
        dbscan.printInfo()
        
        affinity_propagation = AffinityPropagationModel(df)
        affinity_propagation.fit(param.affinity['damping'], param.affinity['preference'])
        affinity_propagation.printInfo()
        
        
        
def plotBestDumping():
        
    metrics = searchBestDumping()
    
    for name, (dados, param) in metrics.items():
        
        dados = dados[['pureza','entropia','silhueta','n_groups']]
        dados.to_csv('Experiments/SearchingDamping/Results_bin/'+param.name+'_results.csv')  
        fig,ax = plt.subplots(figsize=[4,2], dpi=200)
        ax.plot(dados)
        ax.set_title('Searching damping in the '+name+' dataset')
        ax.set_ylabel('Real number of each measure')
        ax.set_xlabel('Damping')
        ax.plot(2*[param.affinity['damping']], [0,3.1], 'k')
        ax.legend(dados.columns.to_list()+['Damping Choosen'],fancybox=True,shadow=True)
        ax.set_xlim([0.49,1])


main()