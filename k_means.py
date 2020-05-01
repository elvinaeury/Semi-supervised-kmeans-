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
    
    k,e=5,10**(-10)
    
    
    #kmeans
    t0=time()
    centers,clusters = kmeans(X,k,e)
    t1=time()
    print('En utilisant kmeans : %f' %(t1-t0))
    
    #comparaison avec kmeans de sklearn
    t2=time()
    resultat = random_kmean_sklearn(k,X)
    t3=time()
    print('En utilisant kmeans de sklearn : %f' %(t3-t2))