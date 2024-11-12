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


X_train=SC.fit_transform(df_train)

X_test=SC.transform(df_test)

X=np.concatenate( (X_train, X_test),axis=0)

 
for attr in bibmon.__all__:             
    a = getattr(bibmon,attr)     
    if isinstance(a, type):         
        if a.__base__ == bibmon._generic_model.GenericModel:   
            if a == bibmon.sklearnManifold:                 
                from sklearn.manifold import TSNE
                model = a(TSNE(n_components=2))
            #else:                    
                #m = a()        
            
                embedded_data=model.fit_transform(X)
                embedded_data.plot_embedding()
                
                
                

            
            
                