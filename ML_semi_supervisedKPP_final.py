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
        unlabelled=rng.rand(sample) < frac
        labels=np.copy(np.zeros(len(clusters)))
        labels[unlabelled]=-1
        X['labelling']=labels # nouvelle colonne avec les 0 (labeled) et -1 (unlabeled)
        
        # base contenant uniquement les labeled data
        # cela évite de travailler sur la base X (qui contient plus de données, donc plus de temps de passer à travers tous le dataframe)
        labelled_base=X[X['labelling']==0].iloc[:,]
        
        # base contenant uniquement les unlabeled data
        unlabelled_base=X[X['labelling']==-1]
        # On enleve les deux colonnes labeling et clusters, pas utile pour la partie 'unlabeled'
        unlabelled_base=unlabelled_base.drop(['labelling','clusters'],axis=1)
        
        
        """ Pour les LABELLED data, on crée les centres pour chaque label"""
        # on détermine les clusters identifiés dans les données labelled
        unique_clusters=labelled_base['clusters'].unique().tolist()
        
        # Le nombre de colonnes est le nombre de features + 2 (pour accomoder les colonnes clusters et labels)
        # le array centers_labelled a le meme nombre de lignes que le nombre de clusters fourni par les données labelled.
        centers_labelled=np.empty((len(unique_clusters),features+2))
        
        for i in range(k):
            # on crée une sous base contenant uniquement les labeled data correspondant aux clusters(1,..,k)
            centers_labelled[i]=labelled_base.loc[labelled_base['clusters'] == i].mean(axis=0)
            # on choisit aléatoirement un centre dans chaque sous-base (tirage sans remise)
#            centers_labeled[i]=f.sample(n=1,replace=False)
            # le centre est obtenu en trouvant la moyenne de chaque colonne (moyenne vérifié)
        centers_labelled=np.delete(centers_labelled,np.s_[39:41], axis=1)

 
        """ Pour les UNLABELLED data, on crée les centres uniquement pour les clusters qui manquent"""
        # on fait appel à la fonction de kmeans ++ sans essaies pour calculer les unlabelled.
        centers_unlabelled=Initialize.kpp_init_notrials(k,unlabelled_base)

        """ On combine les clusters des labelled et unlabelled data"""
        """ On priorise les données labeled, ainsi si C va contenir les centres labeled en premier lieu, 
        puis s'il manque des centres, alors on utilise les centres unlabeled"""
    
        # centers_all va contenir tous les centres, labelled et unlabelled
        centers_all=np.empty((k,features))
        
        K=np.arange(0,k)
        # la liste des clusters qui ne se trouvent PAS dans les labelled.
        not_in_labelled = [i for i in K if i not in unique_clusters]
#        not_in_labelled=[3,4] (test effectué avec une liste fictive < k)
        print(not_in_labelled)
        # On selectionne les centres manquants, des centres de données unlabelled 
        # Il reste vide si aucun centres n'étaient manquants
        selected_from_unlabelled=centers_unlabelled[not_in_labelled,:]
        
        centers_all=np.concatenate((centers_labelled,selected_from_unlabelled),axis=0)
        
        return selected_from_unlabelled,centers_all ,centers_labelled     

        
    
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


class Graph:
        def plot_each(k,X,clusters,centers):
#        centers,clusters=Kmeans.lloyd(k,X,e,centers)
        
            # On affiche les observations, et les couleurs sont basées sur les clusters
            plt.scatter(X.iloc[:,1],X.iloc[:,3],c=clusters,s=7,cmap='viridis')
                
            # On affiche les centres
            plt.scatter(centers[:,1], centers[:,3], marker='*', c='red', s=50)
            
    
        def plot_sklearn(k,X,clusters,centers):
            
            plt.scatter(X.iloc[:,1],X.iloc[:,3],c=clusters,s=7,cmap='viridis')
            plt.scatter(centers[:,1], centers[:,3], marker='o', c='orange', s=50)
        
        
class Display:        
        def main_kpp_NOessaie(k,X,e):
            """ En utilisant l'initialisation kpp, SANS les essaies """
            
            t0=time()
            centers_initial_kpp_notrials=Initialize.kpp_init_notrials(k,X)
            centers_kpp_notrials,centers_kpp_notrials_label=Kmeans.lloyd(k,X,e,centers_initial_kpp_notrials)
            t3=time()
            print('En utilisant kmeans ++ SANS essaie : %f' %(t3-t0))     #   6.0813
            
            return Graph.plot_each(k,X,centers_kpp_notrials_label,centers_kpp_notrials)
        
        
        def main_semisupervised(k,X,e,fraction):
            """ En utilisant l'initialisation kpp, SANS les essaies """
            
            
            t0=time()
            unlabel,cent_all,lab=Initialize.semi_supervised(k,X,fraction)
            lloyd_centers,lloyd_labels=Kmeans.lloyd(k,X,e,cent_all)
            t3=time()
            print('En utilisant kmeans ++ semi supevisée : %f' %(t3-t0))     #   6.0813
            
            return Graph.plot_each(k,X,lloyd_labels,lloyd_centers)
            
        
    
        def main_sklearn(k,X):
            """ En utilisant kmeans_sklearn """
            t0=time()
            centers_sklearn,clusters_sklearn=Kmeans.kmean_sklearn(k,X)
            t4=time()
            print('En utilisant kmeans sklearn : %f' %(t4-t0))
                
            return Graph.plot_sklearn(k,X,clusters_sklearn,centers_sklearn)
        

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
    
#    Display.main_sklearn(k,X)
    
    Display.main_semisupervised(k,X,e,fraction)
    


    