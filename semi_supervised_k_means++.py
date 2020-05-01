#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:02:37 2020

@author: destash
"""

import numpy as np
from others import distance_squared, DataFrameImputer, labeling, kmean_sklearn, plot2D, plot3D
from lloyd import lloyd
import numpy.random as npr
import pandas as pd
from time import time

import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Initialisation des centres pour les kmeans++ semi supervisés
# =============================================================================

def init_ss_kmeans_pp(X_u,X_s,k) : #,L) :
    
    """
    X_u : n_u unlabelled datapoints
    X_s : n_s labeled datapoints
    L : labels corresponding to the data in X_s
    k : number of centers
    """
    
    """initialisation des centres pour les données supervisées"""
    
    centers=X_s.groupby("clusters").mean().to_numpy()
    
    """ajout des centres pour les données non supervisées si on n'a pas encore k centres proportionnellemet à D**2"""
    
    if len(centers) < k :
        distances = distance_squared(centers,X_u)
        dist_min = np.min(distances,axis=0)
        dist_cumul = dist_min.cumsum()
        dist_total = dist_min.sum()
    
    while len(centers) < k :
        random_proba=npr.random()
        random_value=random_proba*dist_total
        
        # On trouve l'indice de l'observation correspondant au random_value
        candidate_id=np.searchsorted(dist_cumul,random_value)
        candidate = np.array([list(X_u.iloc[candidate_id,:])])
        dist_candidate=distance_squared(candidate,X_u)
        
        #actualisation des centres et des distances
        centers=np.concatenate((centers,candidate))
        dist_min = np.minimum(dist_min,dist_candidate)
        dist_cumul = dist_min.cumsum()
        dist_total = dist_min.sum()
        
    return centers

# ================================================================================
# Algorithme des kmeans++ semi supervisés  faisant appel à l'algorithme de Lloyd
# =================================================================================

def ss_kmeans_pp(X,k,e=10**-10) :
    
    """
    X : dataset avec une partie des données déjà labelisées dans une colonne nommée clusters
    k : nombre de clusters
    e : seuil de tolérance
    """
    
    X_s = X[X["clusters"]!=-1]
    X_u = X[X["clusters"]==-1]
    X_u.drop("clusters",axis=1,inplace=True)
    X.drop("clusters",axis=1,inplace=True)
    
    centers = init_ss_kmeans_pp(X_u,X_s,k)
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
    
    # On choisi le pourcentage unlabeled
    fraction=0.4
    
    #labelisation des données
    Y=labeling(k,X,fraction)
    
    #kmeans++ semi supervisés enfin 
    t0=time()
    centers,clusters = ss_kmeans_pp(Y,k,e)
    t1=time()
    print('En utilisant kmeans ++ semi supevisée : %f' %(t1-t0)) #max : 0.185409, min : 0.117940, 4 essais
    
    #comparaison avec kmeans++ sklearn
    t2=time()
    resultat = kmean_sklearn(k,X)
    t3=time()
    print('En utilisant kmeans ++ de sklearn : %f' %(t3-t2)) #max : 5.153863, min : 4.261078, 4 essais
    
    #ss_kmeans_pp environ 30 fois plus rapide sur ces essais avec ces paramètres !!!
    
    #plot kmeans++ semi supervisés
    #plot2D(X,centers,clusters, f"kmeans++ semi supervisés avec {k} clusters")
    plot3D(X,centers,clusters, f"kmeans++ semi supervisés avec {k} clusters")
    
    #plot kmeans++ sklearn
    #plot2D(X,resultat[0],resultat[1],f"kmeans++ de sklearn avec {k} clusters")