#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:51:09 2020

@author: destash
"""

import numpy as np
from copy import deepcopy
from others import distance_squared    

# =============================================================================
# Algorithme de lloyd
# =============================================================================

def lloyd(k,X,centers,e=10**-10):          
        
        """
        k: nombre de clusters
        X: datapoints, in the form of a dataframe
        e: seuil de tolérance, by default 10**(-10)
        centers: initial centers determined by one of the initialisation algorithms
        """
        
        sample,features=X.shape
        
        error=e+1 # On initialise l'erreur: nombre aléatoire > 0
    
        # Assignation des centres
        while error>e:
            distances = distance_squared(centers,X)
                
            # Les clusters sont formés à partir des candidats ayant les distances minimales
            clusters=np.argmin(distances,axis=0)
    
            # Mise a jour des centres
            new_centers=X.groupby(clusters).mean().to_numpy()
    
            error=np.linalg.norm(centers-new_centers)
            
            centers=deepcopy(new_centers)
    
        return centers,clusters