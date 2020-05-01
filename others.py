#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:08:43 2020

@author: destash
"""


from scipy import spatial
from sklearn.cluster import KMeans
import pandas as pd
import  numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA 
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


# =============================================================================
# Distance euclidienne au carré
# =============================================================================
        

def distance_squared(x,y):
    return spatial.distance.cdist(x,y,'sqeuclidean')


# =============================================================================
# kmeans++ de sklearn
# =============================================================================

def kmean_sklearn(k,X):
    k_means = KMeans(n_clusters=k,init="k-means++")
    k_means.fit(X)
    
    centers = k_means.cluster_centers_
    labels=k_means.fit_predict(X)
    
    return centers,labels


# =============================================================================
# kmeans de sklearn
# =============================================================================

def random_kmean_sklearn(k,X):
    k_means = KMeans(n_clusters=k,init="random")
    k_means.fit(X)
    
    centers = k_means.cluster_centers_
    labels=k_means.fit_predict(X)
    
    return centers,labels

def labeling(k,X,frac) :
    
    """
    frac : pourcentage de données non labelisées
    """
    
    labels = kmean_sklearn(k,X)[1]
    nb_unlabeled_data = int(frac*len(labels))
    index_unlabeled_data = np.random.randint(len(labels),size=nb_unlabeled_data)
    labels[index_unlabeled_data]=-1
    X["clusters"] = labels
    
    return X

def ACP(X,centers,nb_comp) :
    pca=PCA(n_components=nb_comp)
    #pca.fit(X) #ACP non centrée réduite
    pca.fit(scale(X)) #ACP centrée réduite
    print(f"Variance cumulée dim {nb_comp} : {100*np.round(pca.explained_variance_ratio_,4).cumsum()[-1]}%")
    PC=pca.transform(X)
    centersPC=pca.transform(centers)
    return PC,centersPC

def plot2D(X,centers,clusters,title) :
    Y=X.to_numpy()
    PC,centersPC = ACP(Y,centers,2)
    plt.scatter(PC[:,0],PC[:,1],c=clusters)
    #plt.scatter(centersPC[:,0],centersPC[:,1],c=np.unique(clusters),marker='*')
    plt.scatter(centersPC[:,0],centersPC[:,1],c='red',marker='*')
    plt.title(title)
    plt.show()
    
def plot3D(X,centers,clusters,title) :
    Y=X.to_numpy()
    PC,centersPC = ACP(Y,centers,3)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure()
    
    ax=fig.add_subplot(221, projection='3d')
    ax.scatter(PC[:,0],PC[:,1],PC[:,2],c=clusters)
    ax.scatter(centersPC[:,0],centersPC[:,1],centersPC[:,2],c='red',marker='*')
    
    ax=fig.add_subplot(222)
    ax.scatter(PC[:,0],PC[:,1],c=clusters)
    ax.scatter(centersPC[:,0],centersPC[:,1],c='red',marker='*')
    
    ax=fig.add_subplot(223)
    ax.scatter(PC[:,0],PC[:,2],c=clusters)
    ax.scatter(centersPC[:,0],centersPC[:,2],c='red',marker='*')
    
    ax=fig.add_subplot(224)
    ax.scatter(PC[:,1],PC[:,2],c=clusters)
    ax.scatter(centersPC[:,1],centersPC[:,2],c='red',marker='*')
    
    plt.title(title)
    plt.show()

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