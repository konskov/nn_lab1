# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:25:01 2019

@author: skovola
"""
import numpy as np

'''only one metric, accuracy
remember to call with y_train.values
because unlike X_train , y_train is not
an np.ndarray'''
def cv_10(classifier_input,X,y):
    length = X.shape[0]
    w_size = (length//10)
    
    classifier = classifier_input()
    
    scores = []
    for k in range(9):
    #31 anti gia 29 deigmata to teleutaio 
        if (k == 0):        
            idx = slice(0,(k+1)*w_size)
        else:
            idx = slice(k*w_size+1,(k+1)*w_size+1)
        
        test_set = X[idx,:]
        train_set = np.delete(X,idx,axis=0)
        test_set_y = y[idx]
        train_set_y = np.delete(y,idx) 
        #now perform classification
        classifier.fit(train_set, train_set_y)
        preds = classifier.predict(test_set)
        acc_list = [int(p == actual) for (p,actual) in zip(preds,test_set_y)]
        acc_score = np.sum(acc_list) / len(preds)   
        scores.append(acc_score)
    
    #last iteration
    idx = slice(9*w_size,length)
    test_set = X[idx,:]    
    train_set = np.delete(X,idx,axis=0)
    test_set_y = y[idx]
    train_set_y = np.delete(y,idx) 
    
    classifier.fit(train_set, train_set_y)
    preds = classifier.predict(test_set)
    acc_list = [int(p == actual) for (p,actual) in zip(preds,test_set_y)]
    acc_score = np.sum(acc_list) / len(preds)   
    scores.append(acc_score)
            
    return scores    



        
        
        
        
        
