# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:07:42 2020

@author: marci
"""

import Config

from sklearn.cluster import KMeans
import numpy as np

class kMeansModel():
    
    # Creates an object with and save all the information that it needs
    # to fit a model of clurstering K-Means
    def __init__(self, data, target_index):
        self.model = None
        self.target = data.iloc[:, target_index]
        self.data = data.drop(axis=1, index = target_index)
    
    ### Primary Functions ###

    def stardartProcess(self, n_clusters):
        # n_clusters: the number of clusters or centroids generated
        # by Kmeans... Try use the number of target/label if exists.
        
        model = KMeans(n_clusters, random_state=Config.seed()).fit(self.data)        
        
        self.model = model
    
    def printInfo(self):
        
        #print('Final locations of the centroid:\n',self.model.cluster_centers_)
        print('number of iterations:  ',self.model.n_iter_, '\n')
        print('lowest SSE value:  ',self.model.inertia_, '\n')
        print('labels:  ',self.model.labels_, '\n')
        num_labels = np.unique(self.model.labels_, return_counts=True)
        print(num_labels)