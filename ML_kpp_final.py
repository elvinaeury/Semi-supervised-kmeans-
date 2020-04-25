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

# =============================================================================
# Les différentes méthodes d'initialisation le k-means
# =============================================================================

class Initialize:
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
        dist=Initialize.distance_squared(np.atleast_2d(centers[0]), X)
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
            dist_candidate=Initialize.distance_squared(np.atleast_2d(X.iloc[candidate_ids,:]),X)
            #Verification de la premiere distance: print(np.sum((X.iloc[54,:]-X.iloc[0,:])**2))
            
            # Choisir le meilleur candidat parmis les 'number_trials' essaies
            # Choisir la distance minimale entre: 
                    # la distance(1er centre et X), et
                    # la distance(k-1 centres restants, pour chaque essaie, et X)
            
            """ Fixer le candidat corespondant à candidate_ids[0] comme le meilleur candidat 
            et le comparer aux autres candidats pour modification si nécessaire """
            best_candidate=candidate_ids[0]
            best_dist=np.minimum(dist_candidate[0],dist)
            best_total_dist=np.sum(best_dist)
    
    
            for essaie in range(1,number_trials):
                new_dist=np.minimum(dist_candidate[essaie],dist)
                new_total_dist=np.sum(new_dist)
                
            # On actualise la plus petite distance et le candidat qui y correspond
                if new_total_dist < best_total_dist :
                    best_candidate=candidate_ids[essaie]
                    best_total_dist=new_total_dist
                    best_dist=new_dist

                    
            # On stock les centres et les distances minimum
            centers[c]=X.iloc[best_candidate]
            
            dist_total=best_total_dist
            dist=best_dist
            dist_cumul=np.cumsum(dist)
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
                
    
    
    def random_init(k,X):
        
        sample,features=X.shape
        centers=np.empty((k,features))

        for i in range(k):
            # tirage sans remise
            centers[i]=X.sample(n=1,replace=False)
        
        return centers
    
    
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
       
    def kmean_sklearn(k,X):
        k_means = KMeans(n_clusters=k,init="k-means++")
        k_means.fit(X)
        
        centers = k_means.cluster_centers_
        labels=k_means.fit_predict(X)
        
        return [centers,labels]
    
    
# =============================================================================
# Affichage des classes et centre
# =============================================================================

class Graph:
    
    def plot_data_all(k,X,e):
        
        init_kpp=Initialize.kpp_init(k,X)
        init_random=Initialize.random_init(k,X)
        init_kpp_notrials=Initialize.kpp_init_notrials(k,X)
    
        centers_kpp,clusters_kpp=Kmeans.lloyd(k,X,e,init_kpp)
        centers_random,clusters_random=Kmeans.lloyd(k,X,e,init_random)
        centers_kpp_notrials,clusters_kpp_notrials=Kmeans.lloyd(k,X,e,init_kpp_notrials)
        centers_sklearn,labels_sklearn=Kmeans.kmean_sklearn(k,X)

        
        # plotting 
        fig, ax = plt.subplots(2, 2,figsize=(5, 5))
        
        ax[0,0].scatter(X.iloc[:,1],X.iloc[:,3],c=clusters_kpp,s=7,cmap='viridis')
        ax[1,0].scatter(X.iloc[:,1],X.iloc[:,3],c=clusters_random,s=7,cmap='viridis')
        ax[0,1].scatter(X.iloc[:,1],X.iloc[:,3],c=clusters_kpp_notrials,s=7,cmap='viridis')
        ax[1,1].scatter(X.iloc[:,1],X.iloc[:,3],c=labels_sklearn,s=7,cmap='viridis')
    
        ax[0,0].scatter(centers_kpp[:,1], centers_kpp[:,3], s=20, c='red', marker="o")
        ax[1,0].scatter(centers_random[:,1], centers_random[:,3], s=55, c='red', marker="X")
        ax[0,1].scatter(centers_kpp_notrials[:,1], centers_kpp_notrials[:,3], s=55, c='red', marker="*")
        ax[1,1].scatter(centers_sklearn[:,1], centers_sklearn[:,3], s=55, c='orange', marker="o")
        
        # Sous-titre
        ax[0,0].set_xlabel('K means ++ AVEC essaie', labelpad = 5)
        ax[1,0].set_xlabel('Initialisation aléatoire', labelpad = 5)
        ax[0,1].set_xlabel('K means ++ SANS essaie', labelpad = 5)
        ax[1,1].set_xlabel('Sklearn', labelpad = 5)
        
        plt.show()
    

    def plot_each(k,X,clusters,centers):
#        centers,clusters=Kmeans.lloyd(k,X,e,centers)
        
        # On affiche les observations, et les couleurs sont basées sur les clusters
        plt.scatter(X.iloc[:,1],X.iloc[:,3],c=clusters,s=7,cmap='viridis')
            
        # On affiche les centres
        plt.scatter(centers[:,1], centers[:,3], marker='*', c='red', s=50)
        
    
    def plot_sklearn(k,X):
        clusters=Kmeans.kmean_sklearn(k,X)
        plt.scatter(X.iloc[:,1],X.iloc[:,3],c=clusters,s=7,cmap='viridis')
        plt.scatter(clusters[:,1], clusters[:,3], marker='*', c='red', s=50)
    
        
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


# =============================================================================
# Rouler les algorithmes
# =============================================================================

class Display:
    
    def main_kpp_essaie(k,X,e):
        """ En utilisant l'initialisation kpp, AVEC les essaies """
        t0=time()
        centers_initial_kpp=Initialize.kpp_init(k,X)
        centers_kpp,centers_kpp_label=Kmeans.lloyd(k,X,e,centers_initial_kpp)
        t1=time()
        print('En utilisant kmeans ++ AVEC essaie : %f' %(t1-t0))
        
        return Graph.plot_each(k,X,centers_kpp_label,centers_kpp)
        

        
    
    def main_random(k,X,e):
        """ En utilisant l'initialisation aléatoire """
        t0=time()
        centers_initial_random=Initialize.random_init(k,X)
        centers_random,centers_random_label=Kmeans.lloyd(k,X,e,centers_initial_random)
        t2=time()
        print('En utilisant kmeans random :  %f' %(t2-t0))
        
        return Graph.plot_each(k,X,centers_random_label,centers_random)
        
    
    
    def main_kpp_NOessaie(k,X,e):
        """ En utilisant l'initialisation kpp, SANS les essaies """
        t0=time()
        centers_initial_kpp_notrials=Initialize.kpp_init_notrials(k,X)
        centers_kpp_notrials,centers_kpp_notrials_label=Kmeans.lloyd(k,X,e,centers_initial_kpp_notrials)
        t3=time()
        print('En utilisant kmeans ++ SANS essaie : %f' %(t3-t0))     #   6.0813
        
        return Graph.plot_each(k,X,centers_kpp_notrials_label,centers_kpp_notrials)
        
    
    
    def main_sklearn(k,X):
        """ En utilisant kmeans_sklearn """
        t0=time()
        centers_sklearn,clusters_sklearn=Kmeans.kmean_sklearn(k,X)
        t4=time()
        print('En utilisant kmeans sklearn : %f' %(t4-t0))
            
        return Graph.plot_each(k,X,clusters_sklearn,centers_sklearn)
        
    
    
    def plot_all(k,X,e):
        return Graph.plot_data_all(k,X,e)
        

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
#    Y.fillna(Y.mean())
    
    k,e=5,10**(-10)
    
    
    """Résultat, en choisir un des cas """
#    Display.main_kpp_essaie(k,X,e)
#    
#    Display.main_random(k,X,e)
#    
#    Display.main_kpp_NOessaie(k,X,e)
#    
#    Display.main_sklearn(k,X)
    
    
    
    Display.plot_all(k,X,e)

