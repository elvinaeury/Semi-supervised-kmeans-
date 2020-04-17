#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:40:19 2020

@author: elvinagovendasamy
"""

# =============================================================================
# Méthode du coude en utilisant deux méthodes
# =============================================================================

class Elbow:

    def kmeans_manual(k,X,e,centers):          
    
        """
        k: clusters
        X: datapoints, in the form of a dataframe
        e: by default 10**(-10)
        initial_centers: determined by one of the initialisation algorithms
        """
        sample,features=X.shape
        
        
        error=e+1 # On initialise l'erreur: nombre aléatoire > 0
        SSE=[]
        sse=0
        sum_distance=0
        # Assignation des centres
        while error>e:
            distances=np.empty((sample,k))
            #sse=np.empty((sample,k))
            clusters=np.empty(features)
    
            for i in range(k):
                distances[:,i]=distance_squared(np.atleast_2d(centers[i]),X)
    
            #Verification de la premiere distance: 
    #        print('verification:',np.sum((centers[0]-X.iloc[0,:])**2))
    #        print(distances)
                
            # Les clusters sont formés à partir des candidats ayant les distances minimales
            clusters=np.argmin(distances,axis=1)
    
            # Mise a jour des centres
            new_centers=deepcopy(centers)
            for i in range(k):
                new_centers[i,:]=np.mean(X.iloc[clusters==i],axis=0)
                sse=spatial.distance.cdist(np.atleast_2d(new_centers),X.iloc[clusters==i],'euclidean')
                #print(sse)
                
            sum_distance=np.sum(sse,axis=1)
            
        #    SSE.append(sum_distance)
        #    print(SSE)
            error=np.linalg.norm(centers-new_centers)
            
            centers=deepcopy(new_centers)
    
        return [centers,clusters,sum_distance]
    
    def kmean_sklearn(k,X):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        
        centers = k_means.cluster_centers_
        labels=k_means.fit_predict(X)
        inertia=k_means.inertia_
        
        return [centers,labels,inertia]
        
        
        
    def elbow_manual(X,e):
        SSE=[]
        k = np.arange(1,5)
        for i in range(1,5):
            centers=random_init(i,X)
            kmeans_inertia=kmeans_manual(i,X,e,centers)[2]
            print(kmeans_inertia)
            SSE.append(kmeans_inertia)
        plt.plot(k, SSE, 'bx-')
    #    plt.xlabel('k')
    #    plt.ylabel('Somme_distance_carre')
    #    plt.title('Méthode du coude')
        plt.show()
        
        
    def elbow_sklearn(X):
        SSE=[]
        k=np.arange(1,5)
        for i in range(1,5):
            kmeans_inertia=kmean_sklearn(i,X)[2]
            print(kmeans_inertia)
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
    Y = data['target']
    
    
    # Nombre de clusters
    k=3
    # Terme d'erreur
    e=10**(-10)
    
    Elbow.elbow_manual(X,e)
    Elbow.elbow_sklearn(X)
