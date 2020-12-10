# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:38 2020

@author: marci
"""

# Seed to be used on every random execution
Seed = 8348

class Parameters:
    
    def __init__(self, name, path, k_clusters, eps, damping):
        
        self.name = name
        self.path = path # Path of the dataset to be read or written
        self.k_clusters = k_clusters # Number of clusters
        self.eps = eps
        self.damping = damping
        
datasets = [
    Parameters('Wine','Data/Wine.csv',3,0.4, 0.5),
    Parameters('Linfoma','Data/Linfoma.csv', 3, 6, 0.5),
    Parameters('Displasia','Data/Displasia.csv', 4, 6, 0.5),
    Parameters('Olivier','Data/OlivierFeatures.csv', 14, 0.4, 0.5),
    #Parameters('Boone','Data/BooneFeatures.csv', 14, 0.4, 0.5),
    Parameters('BreastCancer','Data/BreastCancerWisconsin.csv',2,0.4, 0.5),
    Parameters('Linfoma','Data/Linfoma.csv', 3, 6, 0.5)
]

    
    