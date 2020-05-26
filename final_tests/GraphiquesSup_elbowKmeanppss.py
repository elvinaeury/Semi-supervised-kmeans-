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
from scipy.spatial.distance import cdist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator

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

def SSE(k,X,centers):
    return np.sum(np.min(cdist(X,centers,'euclidean'),axis=1))

def inertie_intra(k,X,e,N,perc):
    distance=0
    for i in range(1,N+1):
        Y=labeling(k,X,perc)
        centers,clusters=ss_kmeans_pp(Y,k,e)
        distance+=SSE(k,Y,centers)
    return distance/N

def elbow_manual(n_clusters,X):
    sample,features=X.shape
    e=10**(-10)

    X = DataFrameImputer().fit_transform(X)    
    SSE=[]
    SSE1=[]
    for i in range(1,n_clusters):
        Y=labeling(i,X,0.6)
        centers,labels=ss_kmeans_pp(Y,i,e)
        centers_sk,labels_sk=kmean_sklearn(i,X)
    
        # en utilisant lloyd
        SSE.append(np.sum(np.min(cdist(X,centers,'euclidean'),axis=1)))
        # en utilisant sklearn
        SSE1.append(np.sum(np.min(cdist(X,centers_sk,'euclidean'),axis=1)))
    
    K=np.arange(1,n_clusters)    
    plt.plot(K,SSE,label='méthode manuel',color='blue')
    plt.plot(K,SSE1,label='méthode sklearn',color='orange')
    plt.xticks(np.arange(1, n_clusters, 1))
    kn = KneeLocator(K, SSE1, curve='convex', direction='decreasing')    
    # plotting dashed_vline on knee
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
    plt.legend()


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
    
    # On choisi le pourcentage unlabeled
    fraction=0.60
    
    #labelisation des données
    Y=labeling(k,quanti_X,fraction)
    
    #kmeans++ semi supervisés 
    t0=time()
    centers,clusters = ss_kmeans_pp(Y,k,e)
    t1=time()
    print('En utilisant kmeans ++ semi supevisée : %f' %(t1-t0)) #max : 0.185409, min : 0.117940, 4 essais
    
 
    
    """ Graphique comparaison des clusters"""
    # Afficher la variable qualitative: shops_used
    
#    sns.lmplot('min_distance_to_shops', 'products_purchased', data=X, hue='shops_used',fit_reg=False,scatter_kws={"s": 10})
#    plt.show()
#    plt.title('Distribution de la variable qualitative: shops_used')
    
    
    # Afficher les clusters avec kmeans++
    
#    X['clusters']=clusters
#    base['clusters']=clusters
#    sns.lmplot('min_distance_to_shops','products_purchased', data=X, hue='clusters',fit_reg=False,scatter_kws={"s": 10},palette=['green','orange','brown','dodgerblue','red'])
##    plt.show()
#    plt.scatter(centers[:,1], centers[:,3], marker='*', c='black', s=50)
    
#    plt.hist(X["avg_price_shop_1"])
#    sns.distplot(X["avg_price_shop_1"],color='skyblue')
#    sns.distplot(X["avg_price_shop_2"], color="orange")
#    sns.boxplot( y=X["avg_price_shop_3"] , color="lavender")
#    sns.boxplot( y=X["avg_price_shop_4"] , color="teal")
#    sns.boxplot( y=X["avg_price_shop_5"] , color="pink")


#    base.groupby('shops_used').count().plot(kind='bar')
#    sns.barplot(x="clusters", y="amount_purchased_shop_1", data=base,palette=['green','orange','brown','dodgerblue','red'])


    """ Graphique methode du coude"""
    coude=elbow_manual(18,quanti_X)

    

#    N=100
#    distance_moyenne=inertie_intra(k,quanti_X,e,N,fraction)
#    print('Inertie_intra en utilisant kmeans++ semi supervisée:', distance_moyenne)
#    
    #ss_kmeans_pp environ 30 fois plus rapide sur ces essais avec ces paramètres !!!
    
    """ ACP """
    #plot kmeans++ semi supervisés
#    plot2D(quanti_X,centers,clusters, f"kmeans++ semi supervisés avec {k} clusters")
#    plot3D(quanti_X,centers,clusters, f"kmeans++ semi supervisés avec {k} clusters")
    
    #plot kmeans++ sklearn
    #plot2D(X,resultat[0],resultat[1],f"kmeans++ de sklearn avec {k} clusters")