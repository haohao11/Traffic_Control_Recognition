# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:55:35 2020
This is the mudule to call:
    the read_junc module for data preprocess
    the data_utils module for loading data
    the model module for classification
@author: cheng
"""

import argparse
import glob
import numpy as np
import os
import sys

from read_junc import preprocess
from data_utils import Load_data

np.set_printoptions(suppress=True)

def parse_args():
    desc = "Tensorflow/keras implementation of RNN LSTM for trajectory prediction"
    parser = argparse.ArgumentParser(description=desc)    
    parser.add_argument('--min_trips', type=int, default=16, 
                        help='minimum number of trips in each junction')
    parser.add_argument('--upper_threshold', type=float, default=100.0, 
                        help='the upper bound distance [m] of each trip to the given junction')
    parser.add_argument('--lower_threshold', type=float, default=10.0, 
                        help='the lower bound distance [m] of each trip to the given junction')
    parser.add_argument('--process_data', type=bool, default=False, 
                        help='the lower bound distance [m] of each trip to the given junction')
    parser.add_argument('--window_size', type=int, default=8, 
                        help='sequence length for the sliding window')
    parser.add_argument('--stride', type=int, default=8, 
                        help='stride for the sliding window')                           
    args = parser.parse_args(sys.argv[1:])
    return args
    

def main():
    
    args = parse_args()
    processeddata_dir = "../Hanover_Dataset/HannoverDataset/processed_data/juncs_trips.npy"
    rawdata_file = "../Hanover_Dataset/HannoverDataset"
    
    # Process and save the data
    if args.process_data:
        junctTrajs_dirs = sorted(glob.glob(
            os.path.join(rawdata_file, "junctTrajs/*.csv")))
        junctions_dir = os.path.join(rawdata_file, "junctions.csv")
           
        juncs_trips = preprocess(junctTrajs_dirs, junctions_dir, 
                                 min_trips=args.min_trips, 
                                 upper_threshold=args.upper_threshold, 
                                 lower_threshold=args.lower_threshold)        
        np.save(processeddata_dir, juncs_trips)
    else:
        juncs_trips = np.load(processeddata_dir)
    
    
    # Load the sequence data
    data_loader = Load_data(juncs_trips, 
                            window_size=args.window_size,
                            stride=args.stride) 
    
    count = 0     
    for sequence, sequence_feature in data_loader.sliding_window():
                
        print(sequence_feature)
        print(sequence)
        
        count += 1       
        if count == 1:
            break





if __name__ == "__main__":
    main()
    
    
    
