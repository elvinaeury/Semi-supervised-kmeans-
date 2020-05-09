#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:48:18 2020

@author: destash
"""

from lloyd import lloyd
import pandas as pd
from others import DataFrameImputer, random_kmean_sklearn
from time import time
from scipy.spatial.distance import cdist
import numpy as np


# =============================================================================
# Initialisation des centres pour les kmeans : tirage uniforme de k centres
# =============================================================================

def init_kmeans(X,k) :
    
    """
    X : dataset
    k : number of centers
    """
    
    centers=X.sample(n=k,replace=False).to_numpy()
        
    return centers


# ================================================================================
# Algorithme des kmeans++ semi supervisés  faisant appel à l'algorithme de Lloyd
# =================================================================================

def kmeans(X,k,e=10**-10) :
    
    """
    X : dataset avec une partie des données déjà labelisées dans une colonne nommée clusters
    k : nombre de clusters
    e : seuil de tolérance
    """
    
    centers = init_kmeans(X,k)
    centers,clusters = lloyd(k,X,centers,e)
    return centers,clusters


def SSE(k,X,centers):
    return np.sum(np.min(cdist(X,centers,'euclidean'),axis=1))

def inertie_intra(k,X,e,N):
    distance=0
    for i in range(1,N+1):
        centers,clusters=kmeans(X,k,e)
        distance+=SSE(k,X,centers)
    return distance/N


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
    quanti_X=X.drop(['shops_used'], axis='columns')
    
    k,e=5,10**(-10)
    
    
    #kmeans
    t0=time()
    centers,clusters = kmeans(quanti_X,k,e)
    t1=time()
    print('En utilisant kmeans : %f' %(t1-t0))
    
    #comparaison avec kmeans de sklearn
    t2=time()
    resultat = random_kmean_sklearn(k,quanti_X)
    t3=time()
    print('En utilisant kmeans de sklearn : %f' %(t3-t2))
    
    N=100
    distance_moyenne=inertie_intra(k,quanti_X,e,N)
    print('Inertie_intra en utilisant kmeans:', distance_moyenne)
    
