# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:08:30 2020

@author: marci
"""

from collections import Counter
from sklearn import metrics
import numpy as np

def purity(labels, target):
    pureza_total = 0
    numero_elementos = len(labels)
    for label_i in np.unique(labels):
        mask = [l == label_i for l in labels]
        target_grupo = target[mask]
        counter = Counter(target_grupo)
        pureza = max(counter.values()) / len(target_grupo)
        
        pureza_total += pureza * len(target_grupo) / numero_elementos
        
    return (pureza_total)

def entropy(labels, target):
    total_entropy = 0
    numero_elementos = len(labels)
    for label_i in np.unique(labels):
        mask = [l == label_i for l in labels]
        target_grupo = target[mask]
        counter = Counter(target_grupo)
        entropy_i = 0
        for t in counter.keys():
            p = counter[t] / len(target_grupo)
            entropy_i -= p * np.log2(p)
            
        
        total_entropy += entropy_i * len(target_grupo) / numero_elementos
    return total_entropy


def printInfo(labels, target, data):
    
    print('purity: ',purity(labels, target))
    print('entropy: ', entropy(labels, target))
    #rand index - [-1,1], quanto maior melhor
    print('rand index: ', metrics.cluster.adjusted_rand_score(target, labels))
    #silhueta - [-1,1], quanto maior, melhor
    print('silhueta: ',metrics.silhouette_score(data, labels))
    #davies bouldin - [>0], quanto menor, melhor
    print('davies-bouldin: ',metrics.davies_bouldin_score(data, labels))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    