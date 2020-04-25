#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:09:13 2020

@author: elvinagovendasamy
"""

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy.random as npr
from copy import deepcopy
from sklearn.cluster import KMeans
import seaborn as sns
from scipy import spatial
from time import time
from sklearn.base import TransformerMixin
from sklearn.semi_supervised import LabelPropagation
from scipy.cluster import hierarchy

# =============================================================================
# Les différentes méthodes d'initialisation le k-means
# =============================================================================

class Initialize:
    
    def distance_squared(x,y):
        return spatial.distance.cdist(x,y,'sqeuclidean')
    

    def kpp_init_notrials(k,X):
        """On crée le premier centre, obtenu aléatoirement de nos observations """
        
        sample,features=X.shape
    
        centers=np.empty((k,features))
        rand=npr.randint(0,sample)
        # le premier centre
        centers[0]=X.iloc[rand,:]
        
        """On obtient les distances de ce premier centre"""
        dist=Initialize.distance_squared(np.atleast_2d(centers[0]), X)
        #Verification de la premiere distance:  print(np.sum(centres[0]-X.iloc[0,:])**2))
        dist_total=np.sum(dist)
        dist_cumul=np.cumsum(dist)
        
        """ On choisi les prochains centres (k-1 restants) en utilisant la probabilité proportionelle à la distance euclidienne au carré"""
        for c in range(1,k):
            random_proba=npr.random() # pas d'essaie dans ce cas, nous n'avons qu'une seule probabilité aléatoire
            random_value=random_proba*dist_total
            # On trouve l'indice de l'observation correspondant au random_value
            candidate_id=np.searchsorted(dist_cumul,random_value)
            
            # Calculer la distance entre le candidat et les observations
            dist_candidate=Initialize.distance_squared(np.atleast_2d(X.iloc[candidate_id,:]),X)
            
            # On calcule la distance minimale entre la distance du candidat et la distance du premier centre
            new_dist=np.minimum(dist_candidate,dist)
            new_total_dist=np.sum(new_dist)
            
            
            # On stock les centres et les distances minimum
            centers[c]=X.iloc[candidate_id]
            
            dist_total=new_total_dist
            dist=new_dist
            dist_cumul = np.cumsum(dist)
        return centers
                
    
    
    def semi_supervised(k,X,frac): # frac doit être entre 0 et 1
        
        sample,features=X.shape
        X = DataFrameImputer().fit_transform(X)
        X.fillna(X.mean())

        k_means = KMeans(n_clusters=k,init="k-means++")
        k_means.fit(X)

        clusters=k_means.fit_predict(X)
        X['clusters']=clusters # nouvelle colonne qui contient les clusters
        
        # on crée des 'labels' pour une proportion donnée d'observations
        rng = np.random.RandomState(sample)
        unlabeled=rng.rand(sample) < frac
        labels=np.copy(np.zeros(len(clusters)))
        labels[unlabeled]=-1
        X['labeling']=labels # nouvelle colonne avec les 0 (labeled) et -1 (unlabeled)
        
        # base contenant uniquement les labeled data
        # cela évite de travailler sur la base X (qui contient plus de données, donc plus de temps de passer à travers tous le dataframe)
        labeled_base=X[X['labeling']==0].iloc[:,]
        
        # base contenant uniquement les unlabeled data
        unlabeled_base=X[X['labeling']==-1]
        # On enleve les deux colonnes labeling et clusters, pas utile pour la partie 'unlabeled'
        unlabeled_base=unlabeled_base.drop(['labeling','clusters'],axis=1)
        
        """ Pour les LABELED data, on crée les centres pour chaque label"""
        centers_labeled=np.empty((k,features))
        for i in range(k):
            # on crée une sous base contenant uniquement les labeled data correspondant aux clusters(1,..,k)
            f=labeled_base.loc[labeled_base['clusters'] == i]
            f=f.drop(['labeling','clusters'],axis=1)
            # on choisit aléatoirement un centre dans chaque sous-base (tirage sans remise)
            centers_labeled[i]=f.sample(n=1,replace=False)
        
        """ Pour les UNLABELED data, on crée les centres pour chaque label"""
        # on fait appel à la fonction de kmeans ++ sans essaies
        centers_unlabeled=Initialize.kpp_init_notrials(k,unlabeled_base)
        
        return centers_labeled,centers_unlabeled
        
    
# =============================================================================
# Algorithme de lloyd
# =============================================================================
        
class Kmeans:

    def lloyd(k,X,e,centers):          
        
        """
        k: clusters
        X: datapoints, in the form of a dataframe
        e: by default 10**(-10)
        initial_centers: determined by one of the initialisation algorithms
        """
        sample,features=X.shape
        
        
        error=e+1 # On initialise l'erreur: nombre aléatoire > 0
    
        # Assignation des centres
        while error>e:
            distances=np.empty((sample,k))
            clusters=np.empty(features)
    
            for i in range(k):
                distances[:,i]=Initialize.distance_squared(np.atleast_2d(centers[i]),X)
            #Verification de la premiere distance: 
    #        print('verification:',np.sum((centers[0]-X.iloc[0,:])**2))
    #        print(distances)
                
            # Les clusters sont formés à partir des candidats ayant les distances minimales
            clusters=np.argmin(distances,axis=1)
    
            # Mise a jour des centres
            new_centers=deepcopy(centers)
            for i in range(k):
                new_centers[i,:]=np.mean(X.iloc[clusters==i],axis=0)
    
            error=np.linalg.norm(centers-new_centers)
            
            centers=deepcopy(new_centers)
    
        return [centers,clusters]
       

    
    
    
        
# =============================================================================
# Enlever les NaNs des données s'il y en a
# =============================================================================

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.
        """
        
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)



class Hierarchy:
    
    def hierar(X):
        return hierarchy.dendrogram(hierarchy.linkage(X,method='ward'))



# =============================================================================
# Main
# =============================================================================

if __name__=="__main__":
    # Uploading ML dataset
    base=pd.read_csv('/Users/elvinagovendasamy/TER/BigML_Dataset.csv',sep=',')
    
    X=base.iloc[:,1:]
    Y=base.iloc[:,0]
    
    # Imputing based on mean for numeric, and most frequent for strings
    X = DataFrameImputer().fit_transform(X)
    X.fillna(X.mean())
    Y.fillna(Y.mean())
    
    k,e=5,10**(-10)
    
    # On choisi le pourcentage unlabeled
    fraction=0.4
    
    semi_supervised_labeled,semi_supervised_unlabeled=Initialize.semi_supervised(k,X,fraction)
