# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:07:42 2020

@author: marci
"""

import Config
import ClusterLabelScore as ls
from sklearn.cluster import KMeans
from sklearn import metrics
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
        
        print('\n\n Dataset: ',name)
        #print('Final locations of the centroid:\n',self.model.cluster_centers_)
        print('number of iterations:  ',self.model.n_iter_, '\n')
        print('SSE value:  ',self.model.inertia_, '\n')
        print('labels:  ',self.model.labels_, '\n')
        labels, amount = np.unique(self.model.labels_, return_counts=True)
        print('Unique labels:\n',labels)
        print('Amount of each on:\n',amount)
        print('purity: ',ls.purity(self.model.labels_, self.target))
        print('entropy: ', ls.entropy(self.model.labels_,self.target))
        #rand index - [-1,1], quanto maior melhor
        print('rand index: ', metrics.cluster.adjusted_rand_score(self.target,self.model.labels_))
        #silhueta - [-1,1], quanto maior, melhor
        print('silhueta: ',metrics.silhouette_score(self.data,self.model.labels_))
        #davies bouldin - [>0], quanto menor, melhor
        print('davies-bouldin: ',metrics.davies_bouldin_score(self.data,self.model.labels_))
