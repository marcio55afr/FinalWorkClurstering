# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:38 2020

@author: marci
"""

# Seed to be used on every random execution
def seed():
    return 8347

class Parameters:
    
    def __init__(self, nome, path, k_clusters):
        
        self.nome = nome
        self.path = path # Path of the dataset to be read or written
        self.k_clusters = k_clusters # Number of clusters
        
datasets = [Parameters('Olivier','Data/OlivierFeatures.csv', 14),
           Parameters('Boone','Data/BooneFeatures.csv', 14),
           Parameters('Linfoma','Data/Linfoma.csv', 3),
           Parameters('Displasia','Data/Displasia.csv', 4)
]

    
    