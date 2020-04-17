#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:18:44 2020

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



def distance_squared(x,y):
    return spatial.distance.cdist(x,y,'sqeuclidean')

def kpp_init(k,X):
    #print(X.shape)
    sample,features=X.shape
    
    """On choisit un nombre d'essaie en ce basant sur les textes/ recherches internet"""
    """ATTENTION: L'utilisation des essaies ne sont pas représentés dans la preuve mathématique mais mentionné dans le texte, nous pourrons l'ajouter comme moyen d'amélioration additionnel"""
    number_trials=2+int(np.log(k)) 
    # Reference:
        # https://github.com/scikit-learn/scikit-learn/pull/99/commits/35ad5ea9ec6dccad6a257a36efba0073f9d83ba2?diff=unified&w=1
        # Conclusion de : https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        
    """On crée le premier centre, obtenu aléatoirement de nos observations """
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
        # Nous créons 'number_trials'(ici 3) nombres de probabilité, la somme n'étant pas égale à 1.
        random_proba=npr.random(number_trials)
        random_value=random_proba*dist_total
        # On trouve l'indice des observations correspondant au random_value
        candidate_ids=np.searchsorted(dist_cumul,random_value)
        
        # Calculer la distance entre ces candidats et les observations
        dist_candidate=distance_squared(np.atleast_2d(X.iloc[candidate_ids,:]),X)
        #Verification de la premiere distance: print(np.sum((X.iloc[54,:]-X.iloc[0,:])**2))
        
        # Choisir le meilleur candidat parmis les 'number_trials' essaies
        # Choisir la distance minimale entre: 
                # la distance(1er centre et X), et
                # la distance(k-1 centres restants, pour chaque essaie, et X)
        
        best_candidate=None
        best_total_dist=None
        best_dist=None
        
        for essaie in range(number_trials):
            new_dist=np.minimum(dist_candidate[essaie],dist)
            new_total_dist=np.sum(new_dist)
            
        # On garde la plus petite distance et le candidat qui y correspond
        if (best_candidate is None) or (new_total_dist<best_total_dist):
            best_candidate=candidate_ids[essaie]
            best_total_dist=new_total_dist
            best_dist=new_dist
            
        # On stock les centres et les distances minimum
        centers[c]=X.iloc[best_candidate]
        
        dist_total=best_total_dist
        dist=best_dist
    #print(centers)   
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
        
        # On garde la distance minimale
        best_candidate=None
        best_total_dist=None
        best_dist=None
        if (best_candidate is None) or (new_total_dist<best_total_dist):
            best_candidate=candidate_id
            best_total_dist=new_total_dist
            best_dist=new_dist
            
        # On stock les centres et les distances minimum
        centers[c]=X.iloc[best_candidate]
        
        dist_total=best_total_dist
        dist=best_dist
    return centers
            




def random_init(k,X):
    
    sample,features=X.shape[0],X.shape[1]
    
    centers=np.empty((k,features))
    # Nous mettons les résultats dans un set afin d'éviter des doublons
    s=set()
    for i in range(k):
        rand=npr.randint(0,sample)
        s.add(rand)
    # On choisit le premier élément du set
    for i in range(len(s)):
        first=list(s)[0]
        s.remove(first)
        centers[i]=X.iloc[first]
    return centers
        


def lloyd(k,X,e,centers):          
    
    """
    k: clusters
    X: datapoints, in the form of a dataframe
    e: by default 10**(-10)
    initial_centers: determined by one of the initialisation algorithms
    """
    sample,features=X.shape[0],X.shape[1]
    
    
    error=10 # On initialise l'erreur: nombre aléatoire > 0
    sse=0
    SSE=[]
    # Assignation des centres
    while error>e:
        distances=np.empty((sample,k))
        clusters=np.empty(features)

        for i in range(k):
            distances[:,i]=distance_squared(np.atleast_2d(centers[i]),X)
            #Verification de la premiere distance: 
#        print(np.sum((centers[0]-X.iloc[0,:])**2))
#        print(distances)
#            
        # Les clusters sont formés à partir des candidats ayant les distances minimales
        clusters=np.argmin(distances,axis=1)

        # Mise a jour des centres
        new_centers=deepcopy(centers)
        for i in range(k):
            new_centers[i,:]=np.mean(X.iloc[clusters==i],axis=0)
        error=np.linalg.norm(centers-new_centers)
        #print(error)
        centers=deepcopy(new_centers)
        sse+=distance_squared(np.atleast_2d(centers),X)
        SSE.append(sse)
        print(SSE.pop())
    
    return [centers,clusters,SSE]
   


def plot_data_all(k,X,e):
    
    init_kpp=kpp_init(k,X)
    init_random=random_init(k,X)
    init_kpp_notrials=kpp_init_notrials(k,X)
    
    #init_all=["init_kpp","init_random","init_kpp_notrials"]

    clusters_kpp=lloyd(k,X,e,init_kpp)[1]
    clusters_random=lloyd(k,X,e,init_random)[1]
    clusters_kpp_notrials=lloyd(k,X,e,init_kpp_notrials)[1]

    centers_kpp=lloyd(k,X,e,init_kpp)[0]
    centers_random=lloyd(k,X,e,init_random)[0]
    centers_kpp_notrials=lloyd(k,X,e,init_kpp_notrials)[0]
    
    
    # plotting 
    fig, ax = plt.subplots(2, 2,figsize=(5, 5))
    
    ax[0,0].scatter(X.iloc[:,1],X.iloc[:,3],c=clusters_kpp,s=7,cmap='viridis')
    ax[1,0].scatter(X.iloc[:,1],X.iloc[:,3],c=clusters_random,s=7,cmap='viridis')
    ax[0,1].scatter(X.iloc[:,1],X.iloc[:,3],c=clusters_kpp_notrials,s=7,cmap='viridis')
    
    ax[0,0].scatter(centers_kpp[:,1], centers_kpp[:,3], s=30, c='black', marker="s")
    ax[1,0].scatter(centers_random[:,1], centers_random[:,3], s=45, c='black', marker="X")
    ax[0,1].scatter(centers_kpp_notrials[:,1], centers_kpp_notrials[:,3], s=50, c='black', marker="*")
    
    # Sous-titre
    ax[0,0].set_xlabel('K means ++ AVEC essaie', labelpad = 5)
    ax[1,0].set_xlabel('Initialisation aléatoire', labelpad = 5)
    ax[0,1].set_xlabel('K means ++ SANS essaie', labelpad = 5)
    
    plt.show()
    
    
    
    
def plot_each(k,X,e,centers):
    
    clusters=lloyd(k,X,e,centers)[1]
    
    # On affiche les observations, et les couleurs sont basées sur les clusters
    plt.scatter(X.iloc[:,1],X.iloc[:,3],c=clusters,s=7,cmap='viridis')
        
    # On affiche les centres
    plt.scatter(centers[:,1], centers[:,3], marker='*', c='black', s=50)
    
        



#****************************************************

if __name__=="__main__":
    iris=datasets.load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])

    X=data.iloc[:,0:4]
    Y = data['target']
    
    
    # Nombre de clusters
    k=3
    # Terme d'erreur
    e=10**(-10)


# =============================================================================
# Initialisation
# =============================================================================
# sample: les observations
# features: les variables
# k: nombre de clusters
    




# =============================================================================
# Rouler l'algorithme de lloyd
# =============================================================================
    
    t0=time()
    
    
# En utilisant l'initialisation kpp, AVEC les essaies
    centers_initial_kpp=kpp_init(k,X)
    centers_kpp=lloyd(k,X,e,centers_initial_kpp)[0]
    t1=time()
    centers_kpp_label=lloyd(k,X,e,centers_initial_kpp)[1]
    Kmeans_inertia=lloyd(k,X,e,centers_initial_kpp)[2]
    print('En utilisant kmeans ++ avec essaie %f' %(t1-t0))    #   0.019


# En utilisant l'initialisation aléatoire
#    centers_initial_random=random_init(k,X)
#    centers_random=lloyd(k,X,e,centers_initial_random)[0]
#    t2=time()
#    centers_random_label=lloyd(k,X,e,centers_initial_random)[1]
#    print('En utilisant kmeans (random) %f' %(t2-t0))          #   0.011327


# En utilisant l'initialisation kpp, SANS les essaies
#    centers_initial_kpp_notrials=kpp_init_notrials(k,X)[0]
#    centers_kpp_notrials=lloyd(k,X,e,centers_initial_kpp_notrials)
#    t3=time()
#    centers_kpp_notrials_label=lloyd(k,X,e,centers_initial_kpp_notrials)[1]
#    print('En utilisant kmeans ++ sans essaie %f' %(t3-t0))     #   0.014018
    

# =============================================================================
# Visualisation: Nuage de points
# =============================================================================
    # Afficher tous les plots
    #graph_all=plot_data_all(k,X,e)
    
    # Afficher chaque plot séparemment
#    graph_kpp=plot_each(k,X,e,centers_initial_kpp)
#    graph_random=plot_each(k,X,e,centers_initial_random)
#    graph_kpp_notrials=plot_each(k,X,e,centers_initial_kpp_notrials)
    
    
        
    
    

    

