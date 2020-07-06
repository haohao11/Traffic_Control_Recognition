# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:07:13 2020

@author: cheng
"""

import numpy as np


class Load_data():
    
    def __init__(self, data, window_size=8, stride=4):
        
        self.data = data
        self.window_size = window_size
        self.stride = stride
                      
        self.ids = np.unique(self.data[:, 0])
        
    
    def sliding_window(self):
        '''
        This is the function to extract the sequence using the sliding window method
        '''
        for id in self.ids:
            trip = self.data[self.data[:, 0]==id]
            trip_feature = self.select_features(trip)
            
            # align the trip and trip_feature
            trip = trip[1:, :]
            
            start = 0
            windsize = self.window_size
            stride = self.stride
            
            while start+windsize <= len(trip):
                seq = trip[start:start+windsize, :]
                seq_feature = trip_feature[start:start+windsize, :]
                start += stride
                # yield seq, seq_feature
                sequence = np.concatenate((seq[:, 0:9], seq_feature), axis=1) 
                yield sequence
                
            
    def select_features(self, trip):
        '''
        Only select the relevent features from each sequence
        Type: numpy
        Header of trip data:
            id, (this is the unique id for the trips across the junctions)
            junc_id,
            junc_arm_rule,
            gid, 
            trip_id, 
            lat, 
            lon, 
            unixtime, 
            timestamp, 
            speed,
            junc_utm_east_to_center
            junc_utm_north_to_center
            junc_utm_to_center
        '''
        # keep only timestamp, speed, junc_utm_east_to_center, junc_utm_east_to_center, junc_utm_to_center
        trip_ = trip[:, -5:]
        _permutation = [0, 4, 2, 3, 1] # move speed to the end
        trip_ = trip_[:, _permutation]
       
        # calculate the offset
        trip_r = trip_[1:, :-1] - trip_[:-1, :-1]
        # calculate the speed in utm_east and utm_north to junction center
        # encountered in true_divide
        
        def speed(trip_data):
            '''
            This is the function to calculate the speed profile by deviding delta_time
            Note: avoid the problem of dividing by zero
            Question: why eixts there zero delta_time?
            '''       
            if np.any((trip_data[:, 0:1] == 0)):
                for i in range(len(trip_data[:, 1:])):
                    if trip_data[i, 0]==0:
                        continue
                    else:
                        trip_data[i, 1:] = trip_data[i, 1:] / trip_data[i, 0]
            else:
                trip_data[:, 1:] = trip_data[:, 1:] / trip_data[:, 0:1]                 
            return trip_data[:, 1:]  
             
        trip_r[:, 1:] = speed(trip_r)
        
        trip_feature = np.concatenate((trip_[1:, 1:], trip_r[:, 1:]), axis=1)
                
        # permuate the feature order to:
        # junc_utm_to_center, utm_east, utm_north, utm_east_speed, utm_east_speed, speed_1, speed_2
        # Why speed_1 and speed_2 are not the same
        permutation = [0, 1, 2, 4, 5, 6, 3]
        
        return trip_feature[:, permutation]
    
    

            
                    
            
        

        
        
        
        
        
        

