# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:55:10 2020

This is the module to use the trip arm_rule for junction arm_rule classification

@author: cheng
"""
import numpy as np

class Junction_class():
    
    def __init__(self, index, ground_truth, prediction):
        self.index = index.reshape(-1, index.shape[-1])
        self.ground_truth = np.argmax(ground_truth.reshape(-1, ground_truth.shape[-1]), axis=1)
        self.prediction = np.argmax(prediction.reshape(-1, ground_truth.shape[-1]), axis=1)
        
    def avg_classfier(self):
        gt = []
        pd = []
               
        self.arm_ids = np.unique(self.index[:, 3]) 
        for arm_id in self.arm_ids:
            arm_index = self.index[:, 3]==arm_id
            arm_ground_truth = self.ground_truth[arm_index]
            arm_prediction = self.prediction[arm_index]
            arm_pd, count = np.unique(arm_prediction, return_counts=True)
            count_percent = count / np.sum(count)
           
            gt.append(arm_ground_truth[0]) # Every step is the same
            pd.append([arm_pd[np.argmax(count_percent)], np.max(count_percent)])
                
        return np.asarray(gt), np.asarray(pd)
    
        
            
            