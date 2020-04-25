#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:40:19 2020

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
from scipy.spatial.distance import cdist
from kneed import KneeLocator


# =============================================================================
# Méthode du coude en utilisant deux méthodes
# =============================================================================
# Remove any NA
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

    
def distance_squared(x,y):
    return spatial.distance.cdist(x,y,'sqeuclidean')


def random_init(k,X):
    
    sample,features=X.shape
    centers=np.empty((k,features))

    for i in range(k):
        # tirage sans remise
        centers[i]=X.sample(n=1,replace=False)
        
    return centers

def kpp_init_notrials(k,X):
        """On crée le premier centre, obtenu aléatoirement de nos observations """
        
        sample,features=X.shape
    
        centers=np.empty((k,features))
        rand=npr.randint(0,sample)
        # le premier centre
        centers[0]=X.iloc[rand,:]
        
        """On obtient les distances de ce premier centre"""
        dist=distance_squared(np.atleast_2d(centers[0]), X)
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
            dist_candidate=distance_squared(np.atleast_2d(X.iloc[candidate_id,:]),X)
            
            # On calcule la distance minimale entre la distance du candidat et la distance du premier centre
            new_dist=np.minimum(dist_candidate,dist)
            new_total_dist=np.sum(new_dist)
            
            
            # On stock les centres et les distances minimum
            centers[c]=X.iloc[candidate_id]
#            print(X.iloc[candidate_id])
            
            dist_total=new_total_dist
            dist=new_dist
            dist_cumul = np.cumsum(dist)
        return centers



# =============================================================================
# Algorithme de lloyd
# =============================================================================

def lloyd(k,X,e,centers):          
    
    """
    k: clusters
    X: datapoints, in the form of a dataframe
    e: by default 10**(-10)
    initial_centers: determined by one of the initialisation algorithms
    """
    sample,features=X.shape
    
#    print(centers)
    error=e+1 # On initialise l'erreur: nombre aléatoire > 0

    # Assignation des centres
    while error>e:
        distances=np.empty((sample,k))
        clusters=np.empty(features)

        for i in range(k):
            distances[:,i]=distance_squared(np.atleast_2d(centers[i]),X)

        # Les clusters sont formés à partir des candidats ayant les distances minimales
        clusters=np.argmin(distances,axis=1)

        # Mise a jour des centres
        new_centers=deepcopy(centers)
        for i in range(k):
            new_centers[i,:]=np.mean(X.iloc[clusters==i],axis=0)
            print(X.iloc[clusters==i])
        error=np.linalg.norm(centers-new_centers)
        
        centers=deepcopy(new_centers)
        
    return [centers,clusters]


def elbow_manual(n_clusters,X):
    sample,features=X.shape
    e=10**(-10)
    # ici on force l'initialisation aléatoire
    initial_centers=kpp_init_notrials(n_clusters,X)
#    initial_random=random_init(n_clusters,X)
    X = DataFrameImputer().fit_transform(X)
#    X.fillna(X.mean())
    
    SSE=[]
    SSE1=[]
    for i in range(1,n_clusters):
        centers,labels=lloyd(i,X,e,initial_centers)
        centers_sk,labels_sk=kmean_sklearn(i,X)
    
        SSE.append(np.sum(np.min(cdist(X,centers,'euclidean'),axis=1)))
        SSE1.append(np.sum(np.min(cdist(X,centers_sk,'euclidean'),axis=1)))
    
    K=np.arange(1,n_clusters)    
    plt.plot(K,SSE)
    plt.title('Méthode du coude')
    plt.show()
    
#    # On doit prendre au minimum 2 clusters
#    K_=np.arange(2,n_clusters)
#    kn = KneeLocator(K_, SSE, curve='convex', direction='decreasing')    
#    print(kn.knee)
#    # plotting dashed_vline on knee
#    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

# =============================================================================
# Sklearn elbow method
# =============================================================================

def kmean_sklearn(k,X):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    
    centers = k_means.cluster_centers_
    labels=k_means.fit_predict(X)
    
    
    return [centers,labels]
    
    
    
def elbow_sklearn(n_clusters,X):
    SSE=[]
    k=np.arange(1,n_clusters)
    for i in range(1,n_clusters):
        k_means = KMeans(i)
        k_means.fit(X)
        kmeans_inertia=k_means.inertia_
        
        SSE.append(kmeans_inertia)
    plt.plot(k, SSE, 'bx-')
    plt.show()
    kn = KneeLocator(k, SSE, curve='convex', direction='decreasing')    
    print(kn.knee)
    # plotting dashed_vline on knee
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')




# =============================================================================
# Main
# =============================================================================

if __name__=="__main__":
    
    base=pd.read_csv('/Users/elvinagovendasamy/TER/BigML_Dataset.csv',sep=',')
    
    X=base.iloc[:,1:]
    Y=base.iloc[:,0]
    
    # Imputing based on mean for numeric, and most frequent for strings
    X = DataFrameImputer().fit_transform(X)
    X.fillna(X.mean())
#    Y.fillna(Y.mean())
    
    k,e=5,10**(-10)
    
    n_clusters=7
    
#    kpp=kpp_init_notrials(k,X)
#    random_=random_init(k,X)
#    lloyd_=lloyd(k,X,e,kpp)
    
    
    graph1=elbow_manual(n_clusters,X)
#    elbow_sklearn(n_clusters,X)
