# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:02:59 2020

@author: cheng
"""
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report


class Evaluation():
    
    def __init__(self, ground_truth, prediction, target_names=None):
        self.ground_truth = np.argmax(ground_truth, axis=1)
        self.prediction = np.argmax(prediction, axis=1)
        self.target_names = target_names
        
        
    def cf_matrix(self):
        matrix = confusion_matrix(self.ground_truth, 
                                  self.prediction)
        print(matrix)
        
    
    def report(self):
        print(classification_report(self.ground_truth, 
                              self.prediction,
                              target_names=self.target_names))
        
    
    
    
        
        
    


