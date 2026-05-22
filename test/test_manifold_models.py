# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:34:01 2024

@author: Leonardo Voltolini
"""

import bibmon
from sklearn.preprocessing import StandardScaler
import numpy as np



SC=StandardScaler()

# loading the data from TEP
df_train, df_test = bibmon.load_tennessee_eastman(train_id = 0, 
                                                   test_id = 1)

#Transforming training and testing data using StandardScaler
X_train=SC.fit_transform(df_train)
X_test=SC.transform(df_test)

#Concatenating train and the test, because manifold models normally
#don't require a separation between training and testing folds
X=np.concatenate( (X_train, X_test),axis=0)

 
for attr in bibmon.__all__:             
    a = getattr(bibmon,attr)     
    if isinstance(a, type):         
        '''
        Verifying if the attribute a is generic model from sklearn manifold
        and then applying the adequate model as wanted
        '''
        if a.__base__ == bibmon._generic_model.GenericModel:   
            if a == bibmon.sklearnManifold:             
                from sklearn.manifold import TSNE
                model = a(TSNE(n_components=2)) #Creating the model
                
                '''
                Computing the embeeding data from fit_transform function
                and subsequently plotting the clustering in the appropriate 
                dimension
                '''
                embedded_data=model.fit_transform(X)
                model.plot_embedding()
    
#%%

'''
This implementation does the same as previous cell, but, it applies
a distinct model and automatically computes fit_transform and clusters 
visualization
'''

for attr in bibmon.__all__:             
    a = getattr(bibmon,attr)     
    if isinstance(a, type):         
        if a.__base__ == bibmon._generic_model.GenericModel:   
            if a == bibmon.sklearnManifold:                 
                from sklearn.manifold import MDS
                model = a(MDS(n_components=3))
                
                '''
                The below code transforms the data and presents the
                graph for cluster visualization
                '''
                model.clusters_visualization(X)
                
                
                
                
                

            
            
                