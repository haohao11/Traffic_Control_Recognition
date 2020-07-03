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

from data_utils import Load_data
from read_junc import preprocess
from model import CVAE


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
    parser.add_argument('--num_features', type=int, default=7, 
                        help='number of input features')
    parser.add_argument('--num_classes', type=int, default=5, 
                        help='number of input classes')

    parser.add_argument('--encoder_dim', type=int, default=16, 
                        help='This is the size of the encoder output dimension')
    parser.add_argument('--z_dim', type=int, default=2, 
                        help='This is the size of the z dimension')
    parser.add_argument('--z_decoder_dim', type=int, default=64, 
                        help='This is the size of the decoder LSTM dimension')
    parser.add_argument('--hidden_size', type=int, default=32, 
                        help='The size of GRU hidden state')
    parser.add_argument('--batch_size', type=int, default=352, help='Batch size')
    parser.add_argument('--s_drop', type=float, default=0.1, 
                        help='The dropout rate for trajectory sequence')
    parser.add_argument('--z_drop', type=float, default=0.2, 
                        help='The dropout rate for z input')
    parser.add_argument('--beta', type=float, default=0.75, 
                        help='Loss weight')   
    parser.add_argument('--train_mode', type=bool, default=True, 
                        help='This is the training mode')
    parser.add_argument('--split', type=float, default=0.8, 
                        help='the split rate for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--aug_num', type=int, default=8, 
                        help='Number of augmentations')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of batches')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Maximum mumber of continuous epochs without converging')    


                          
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
        
        r = sequence_feature[:, -1] / sequence_feature[:, -2]
        print(r)
        
        count += 1       
        if count == 1:
            break
        
    # # Instantiate the model
    cvae = CVAE(args)   
    # Contruct the cave model    
    train =cvae.training() 
    train.summary() 
    
    x_encoder=cvae.X_encoder()
    decoder = cvae.Decoder()       
    x_encoder.summary()
    decoder.summary()
    





if __name__ == "__main__":
    main()
    
    
    
