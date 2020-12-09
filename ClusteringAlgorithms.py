# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:07:42 2020

@author: marci
"""

import Config
import ClusterLabelScore as ls
from sklearn.cluster import KMeans, MeanShift, DBSCAN,SpectralClustering, AffinityPropagation
import numpy as np

class Model():
     
    # Save all information that it needs to fit a model
    def __init__(self, data):
        self.model = None
        self.target = data['target']
        self.data = data.drop(columns = 'target')
        
    
    def printInfo(self):
        
        #print('Final locations of the centroid:\n',self.model.cluster_centers_)
        labels, amount = np.unique(self.model.labels_, return_counts=True)
        ls.printInfo(self.model.labels_, self.target, self.data)
        print('Unique labels:\n',labels)
        print('Amount of each on:\n',amount,end='\n\n')
        
    def preProcessing_(self):
        self.data = (self.data - self.data.mean())/self.data.std() 
    
    def getValidation(self):
        return (ls.getValidation(self.model.labels_, self.target, self.data))
        

class KMeansModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
    
    # Runs the cluresting using the K-Means algorithm 
    def fit(self, n_clusters):
        # n_clusters: the number of clusters or centroids generated
        # by Kmeans... Try use the number of target/label if exists.
        
        model = KMeans(n_clusters, random_state=Config.Seed).fit(self.data)        
        
        self.model = model

    def printInfo(self):
        print('\nInfomation about K-Means model\n')
        print('SSE value:  ',self.model.inertia_)
        print('number of iterations:  ',self.model.n_iter_)
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
        print('number of iterations:  ',self.model.n_iter_)
        super().printInfo()
        
  

class DBSCANModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
    
    # Runs the cluresting using the Mean shift algorithm 
    def fit(self, eps):
        
        model = DBSCAN(eps = eps, min_samples=5).fit(self.data) 
        self.model = model

    def printInfo(self):
        print('\nInfomation about DBSCAN model\n')
        super().printInfo()
        
class SpectralClusteringModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
    
    # Runs the cluresting using the Mean shift algorithm 
    def fit(self,k_clusters):
        
        model = SpectralClustering(n_clusters=k_clusters,
                                   assign_labels="discretize",
                                   random_state=Config.Seed).fit(self.data)
        self.model = model

    def printInfo(self):
        print(self.model.labels_)
        print('\nInfomation about SpectralClustering model\n')
        super().printInfo()

class AffinityPropagationModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
    
    # Runs the cluresting using the Mean shift algorithm 
    def fit(self, damping = 0.5):
        
        model = AffinityPropagation(damping = damping,
                                    random_state=Config.Seed).fit(self.data)
        self.model = model

    def printInfo(self):
        print(self.model.labels_)
        print('\nInfomation about Affinity Propagation model\n')
        super().printInfo()

'''
    # Didn' work, dbscan is clurstering all data as one cluster
    # and some information cannot calculate that.
class OpticsModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
        self.method = 'xi' # default method of the OPTICS object
    
    # Runs the cluresting using the OPTICS algorithm 
    def fit(self):
        
        model = OPTICS().fit(self.data) 
        self.model = model
        
    def fit_dbscan(self):
        
        model = OPTICS(cluster_method='dbscan',max_eps=1).fit(self.data)
        self.method = 'dbscan'
        self.model = model

    def printInfo(self):
        print('\nInfomation about Optics model using the method ' + self.method, end='\n\n')
        super().printInfo()
      
 
       
# Just need three different model on this work

class GaussianMixtureModel(Model):
    
    # Creates an object with data and its super Inherence 
    def __init__(self, data):
        super().__init__(data)
        self.labels = None
        
    # Runs the cluresting using the OPTICS algorithm 
    def fit(self, num_labels):            
        self.labels = GaussianMixture(n_components=num_labels, covariance_type='full').fit_predict(self.data)

    def printInfo(self):
        print('\nInfomation about Gaussian Mixture model\n')
        labels, amount = np.unique(self.labels, return_counts=True)
        ls.printInfo(self.labels, self.target, self.data)
        print('Unique labels:\n',labels)
        print('Amount of each on:\n',amount,end='\n\n')
        
'''       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        