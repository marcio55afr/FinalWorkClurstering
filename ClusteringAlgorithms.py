# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:07:42 2020

@author: marci
"""

import Config
import ClusterLabelScore as ls
from sklearn.cluster import KMeans
import numpy as np

class kMeansModel():
    
    # Creates an object with and save all the information that it needs
    # to fit a model of clurstering K-Means
    def __init__(self, data):
        self.model = None
        self.target = data['target']
        self.data = data.drop(columns = 'target')
    
    ### Primary Functions ###

    def stardartProcess(self, n_clusters):
        # n_clusters: the number of clusters or centroids generated
        # by Kmeans... Try use the number of target/label if exists.
        
        model = KMeans(n_clusters, random_state=Config.seed()).fit(self.data)        
        
        self.model = model
    
    def printInfo(self, name):
        
        print('\n\n\n\n\tDataset: ',name)
        #print('Final locations of the centroid:\n',self.model.cluster_centers_)
        #print('labels:  ',self.model.labels_, '\n')
        labels, amount = np.unique(self.model.labels_, return_counts=True)
        print('\nUnique labels:\n',labels)
        print('Amount of each on:\n',amount,end='\n\n')
        print('number of iterations:  ',self.model.n_iter_)
        print('SSE value:  ',self.model.inertia_)
        ls.printInfo(self.model.labels_, self.target, self.data)
