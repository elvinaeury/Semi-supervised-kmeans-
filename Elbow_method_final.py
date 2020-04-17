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


# =============================================================================
# Méthode du coude en utilisant deux méthodes
# =============================================================================



class Elbow:

    def kmeans_manual(k,X,e):          
    
        """
        k: clusters
        X: datapoints, in the form of a dataframe
        e: by default 10**(-10)
        initial_centers: determined by one of the initialisation algorithms
        """
        sample,features=X.shape
# =============================================================================
# Initialisation random
# =============================================================================
        centers=np.empty((k,features))
       
        rands_deja_obtenus = set()
        for i in range(k):
            n = len(rands_deja_obtenus)
            while len(rands_deja_obtenus) == n :
                rand=npr.randint(0,sample)
                rands_deja_obtenus.add(rand)
            centers[i]=X.iloc[rand]
# =============================================================================
# Lloyd
# =============================================================================           
        error=e+1 
        sse=0
        sum_distance=0
        # Assignation des centres
        while error>e:
            distances=np.empty((sample,k))
            #sse=np.empty((sample,k))
            clusters=np.empty(features)
    
            for i in range(k):
                distances[:,i]=spatial.distance.cdist(np.atleast_2d(centers[i]),X,'sqeuclidean')

            clusters=np.argmin(distances,axis=1)
    
            # Mise a jour des centres
            new_centers=deepcopy(centers)
            for i in range(k):
                new_centers[i,:]=np.mean(X.iloc[clusters==i],axis=0)
                sse=spatial.distance.cdist(np.atleast_2d(new_centers),X.iloc[clusters==i],'sqeuclidean')
                print(sse)
                sum_distance=np.sum(sse,axis=1)
                
            
            error=np.linalg.norm(centers-new_centers)
            
            print(sum_distance)
            
            centers=deepcopy(new_centers)
    
        return [centers,clusters,sum_distance]
    
    
    def kmean_sklearn(k,X):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        
        centers = k_means.cluster_centers_
        labels=k_means.fit_predict(X)
        inertia=k_means.inertia_
        print(inertia)
        
        return [centers,labels,inertia]
        
        
        
    def elbow_manual(X,e):
        k = np.arange(1,5)
        for i in range(1,5):
            kmeans_inertia=Elbow.kmeans_manual(i,X,e)[2]
        plt.plot(k, kmeans_inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Somme_distance_carre')
        plt.title('Méthode du coude')
        plt.show()
        
        
    def elbow_sklearn(X):
        SSE=[]
        k=np.arange(1,5)
        for i in range(1,5):
            kmeans_inertia=Elbow.kmean_sklearn(i,X)[2]
            SSE.append(kmeans_inertia)
        plt.plot(k, SSE, 'bx-')
        plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__=="__main__":
    
    iris=datasets.load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])

    X=data.iloc[:,0:4]
#    Y = data['target']
    
    k,e=3,10**(-10)
    
    
    

    
#    Elbow.elbow_manual(X,e)
    Elbow.elbow_sklearn(X)
