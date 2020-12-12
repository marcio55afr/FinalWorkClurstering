# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:38 2020

@author: marci
"""

# Seed to be used on every random execution
Seed = 8348

class Parameters:
    
    def __init__(self, name, path, k_clusters, eps, affinity = {}):
        
        self.name = name
        self.path = path # Path of the dataset to be read or written
        self.k_clusters = k_clusters # Number of clusters
        self.eps = eps
        self.affinity = affinity
        
datasets = [
    Parameters('Wine','Data/Wine.csv',3,0.4, 
                                    affinity = {'damping': 0.74,
                                                'preference': -300}),
    Parameters('Linfoma','Data/Linfoma.csv', 3, 6,  
                                    affinity = {'damping': 0.74,
                                                'preference': -3000}),
    Parameters('Displasia','Data/Displasia.csv', 4, 6, 
                                    affinity = {'damping': 0.86,
                                                'preference': -3000}),
    Parameters('BreastCancer','Data/BreastCancerWisconsin.csv',2,0.4,  
                                    affinity = {'damping': 0.88,
                                                'preference': -1740}),
    #Parameters('Olivier','Data/OlivierFeatures.csv', 14, 0.4,  
    #                                affinity = {'damping': 0.82,
    #                                            'preference': -3550}),
    Parameters('Boone','Data/BooneFeatures.csv', 14, 0.4,  
                                    affinity = {'damping': 0.74,
                                                'preference': -3000})
]

    
    