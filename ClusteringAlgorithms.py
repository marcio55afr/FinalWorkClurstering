# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:07:42 2020

@author: marci
"""

import Config
import ClusterLabelScore as ls
from sklearn.cluster import KMeans, MeanShift
import numpy as np

class Model():
     
    # Save all information that it needs to fit a model
    def __init__(self, data):
        self.model = None
        self.target = data['target']
        self.data = data.drop(columns = 'target')
        
    
    def printInfo(self):
        
        #print('Final locations of the centroid:\n',self.model.cluster_centers_)
        #print('labels:  ',self.model.labels_, '\n')
        labels, amount = np.unique(self.model.labels_, return_counts=True)
        ls.printInfo(self.model.labels_, self.target, self.data)
        print('number of iterations:  ',self.model.n_iter_)
        print('Unique labels:\n',labels)
        print('Amount of each on:\n',amount,end='\n\n')
        
        

class KMeansModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
    
    # Runs the cluresting using the K-Means algorithm 
    def fit(self, n_clusters):
        # n_clusters: the number of clusters or centroids generated
        # by Kmeans... Try use the number of target/label if exists.
        
        model = KMeans(n_clusters, random_state=Config.seed()).fit(self.data)        
        
        self.model = model

    def printInfo(self):
        print('\nInfomation about K-Means model\n')
        print('SSE value:  ',self.model.inertia_)
        super().printInfo()        
        
        
        
        

class MeanShiftModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
    
    # Runs the cluresting using the Mean shift algorithm 
    def fit(self):
        
        model = MeanShift().fit(self.data) 
        self.model = model

    def printInfo(self):
        print('\nInfomation about Mean Shift model\n')
        super().printInfo()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        