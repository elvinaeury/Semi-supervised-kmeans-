#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:28:40 2020

@author: destash
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:51:49 2020

@author: destash
"""

import numpy as np
from copy import deepcopy
from others import distance_squared, DataFrameImputer #, kmean_sklearn
import numpy.random as npr
import pandas as pd
#from time import time
import matplotlib.pyplot as plt

# =============================================================================
# Algorithme de lloyd modifié retourant en plus l'inertie intra
# =============================================================================

def lloyd(k,X,centers,e=10**-10):          
        
        """
        k: nombre de clusters
        X: datapoints, in the form of a dataframe
        e: seuil de tolérance, by default 10**(-10)
        centers: initial centers determined by one of the initialisation algorithms
        """
        
        sample,features=X.shape
        
        error=e+1 # On initialise l'erreur: nombre aléatoire > 0
    
        # Assignation des centres
        count=0
        while error>e:
            count += 1
            distances = distance_squared(centers,X)
            #print(distances.shape)
            #print("count : ", count)
                
            # Les clusters sont formés à partir des candidats ayant les distances minimales
            dist_min=np.min(distances,axis=0)
            inertie=np.sum(dist_min)
            #print("inertie : ",inertie)
            clusters=np.argmin(distances,axis=0)
    
            # Mise a jour des centres
            new_centers=X.groupby(clusters).mean().to_numpy()
    
            error=np.linalg.norm(centers-new_centers)
            
            centers=deepcopy(new_centers)
    
        return centers,clusters,inertie

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
    centers,clusters,inertie = lloyd(k,X,centers,e)
    
    return centers,clusters,inertie


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
    
    k,e=20,10**(-10)
    
    liste_inertie = []
    for i in range(1,k+1) :
        inertie = kmeans_pp(X,i)[2]
        liste_inertie.append(inertie)
        
    plt.plot(range(1,k+1),liste_inertie)
    
    
    
