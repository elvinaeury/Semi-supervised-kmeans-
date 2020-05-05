#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:51:49 2020

@author: destash
"""

import numpy as np
import pandas as pd
from others import distance_squared, DataFrameImputer #, kmean_sklearn
from k_means_pp import kmeans_pp #k_means_pp = kmeans++.py
import matplotlib.pyplot as plt

# =============================================================================
# Méthode du coude basée sur kmeans++
# =============================================================================

def coude(X,K=8,e=10**-10):
    
    """
    K : sup des valeurs de k
    X : dataset
    e : seuil de tolérance
   """
   
    liste_inertie = []
    for i in range(1,K+1) :
        centers = kmeans_pp(X,i,e)[0]
        distances = distance_squared(centers,X)
        dist_min=np.min(distances,axis=0)
        inertie=np.sum(dist_min)
        liste_inertie.append(inertie)
        
    plt.plot(range(1,K+1),liste_inertie)
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__=="__main__":
    # Uploading ML dataset
    base=pd.read_csv('BigML_Dataset.csv',sep=',')
    
    X=base.iloc[:,1:]
    
    # Imputing based on mean for numeric, and most frequent for strings
    X = DataFrameImputer().fit_transform(X)
    X.fillna(X.mean())
    
    K=20
    coude(X,K)
