#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:51:49 2020

@author: destash
"""

import numpy as np
from others import distance_squared, DataFrameImputer, kmean_sklearn
from lloyd import lloyd
import numpy.random as npr
import pandas as pd
from time import time
from scipy.spatial.distance import cdist

# =============================================================================
# Initialisation des centres pour les kmeans++
# =============================================================================

def init_kmeans_pp(X,k) :
    
    """
    X : dataset
    k : number of centers
    """
    
    """On crée le premier centre, obtenu aléatoirement de nos observations """
    sample = X.shape[0]
    rand=npr.randint(0,sample)
    centers=np.array([list(X.iloc[rand,:])]) #array 2D avec un seul centre ici
    
    """On obtient les distances à ce premier centre"""
    dist_min = distance_squared(centers,X)
    dist_cumul = dist_min.cumsum()
    dist_total = dist_min.sum()
    
    """ On choisit les prochains centres (k-1 restants) en utilisant la probabilité proportionelle à la distance euclidienne au carré"""
    while len(centers) < k :
        random_proba=npr.random()
        random_value=random_proba*dist_total
        
        # On trouve l'observation correspondant au random_value et on calcule les distances à cette observation
        candidate_id=np.searchsorted(dist_cumul,random_value)
        candidate = np.array([list(X.iloc[candidate_id,:])]) #array 2D avec un seul centre
        dist_candidate=distance_squared(candidate,X)
        
        #actualisation des centres et des distances
        centers=np.concatenate((centers,candidate))
        dist_min = np.minimum(dist_min,dist_candidate)
        dist_cumul = dist_min.cumsum()
        dist_total = dist_min.sum()
        
    return centers

# ================================================================================
# Algorithme des kmeans++ semi supervisés  faisant appel à l'algorithme de Lloyd
# =================================================================================

def kmeans_pp(X,k,e=10**-10) :
    
    """
    X : dataset avec une partie des données déjà labelisées dans une colonne nommée clusters
    k : nombre de clusters
    e : seuil de tolérance
    """
    
    centers = init_kmeans_pp(X,k)
    centers,clusters = lloyd(k,X,centers,e)
    
    return centers,clusters

def SSE(k,X,centers):
    return np.sum(np.min(cdist(X,centers,'euclidean'),axis=1))

def inertie_intra(k,X,e,N):
    distance=0
    for i in range(1,N+1):
        centers,clusters=kmeans_pp(X,k,e)
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
    
    
    #kmeans++
    t0=time()
    centers,clusters = kmeans_pp(quanti_X,k,e)
    t1=time()
    print('En utilisant kmeans++ : %f' %(t1-t0))
    
    #comparaison avec kmeans de sklearn
    t2=time()
    resultat = kmean_sklearn(k,quanti_X)
    t3=time()
    print('En utilisant kmeans++ de sklearn : %f' %(t3-t2))
    

    N=100
    distance_moyenne=inertie_intra(k,quanti_X,e,N)
    print('Inertie_intra en utilisant kmeans++:', distance_moyenne)