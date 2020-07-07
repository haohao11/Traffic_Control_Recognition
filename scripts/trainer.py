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
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler


from data_utils import Load_data
from evaluation import Evaluation
from model import CVAE
from read_junc import preprocess



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
    parser.add_argument('--num_classes', type=int, default=3, 
                        help='number of input classes')

    parser.add_argument('--encoder_dim', type=int, default=128, 
                        help='This is the size of the encoder output dimension')
    parser.add_argument('--z_dim', type=int, default=2, 
                        help='This is the size of the z dimension')
    parser.add_argument('--z_decoder_dim', type=int, default=128, 
                        help='This is the size of the decoder LSTM dimension')
    parser.add_argument('--hidden_size', type=int, default=128, 
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
    parser.add_argument('--split', type=float, default=0.7, 
                        help='the split rate for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--aug_num', type=int, default=8, 
                        help='Number of augmentations')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Maximum mumber of continuous epochs without converging')    
                          
    args = parser.parse_args(sys.argv[1:])
    return args


def normalization(data):
    '''
    MinMax normalization
    '''
    scaler = MinMaxScaler()
    print(scaler.fit(data))
    print(scaler.data_max_)
    normal_feature = scaler.transform(data)
    return normal_feature
    

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
    
    
    # Load the sequence data using the sliding window scheme
    data_loader = Load_data(juncs_trips, 
                            window_size=args.window_size,
                            stride=args.stride) 
        
    # Load the sequence data and get the data index
    data = [sequence for sequence in data_loader.sliding_window()]
    data = np.reshape(data, (-1, 16)) # data index + features = 16
    
    # Note, due to the data imbalance, 
    # merge tram_rails (-1) and yield_sigh (1) to priority_sign (2)
    if args.num_classes==3:
        data[data[:, 2]==-1, :] = 2
        data[data[:, 2]==1, :] = 2
        
        # new target class: 
        # uncontrolled:0, 
        # traffic_light:1, 
        # tram_rails/yield_sigh/priority_sign
        data[data[:, 2]==4, :] = 1
           
    # Normalize the features
    data[:, 9:] = normalization(data[:, 9:])
    
    # Get the class label
    label = data[:, 2].astype(int)
    assert args.num_classes== len(np.unique(label)), "The number of classes is not correct"
    _label = np.eye(args.num_classes)[label].reshape(-1, args.window_size, args.num_classes)    
        
    # Question: how to do the data partitioning    
    data = np.reshape(data, (-1, args.window_size, 16))
    print(data.shape)
    
    # train_val_split = data[:, 0, 1]<5000    
    train_val_split = np.random.rand(len(data)) < args.split
    
    train_data_index = data[train_val_split, :, :9]
    train_x = data[train_val_split, :, 9:]
    train_y = _label[train_val_split, :, :]
    
    val_data_index = data[~train_val_split, :, :9]
    val_x = data[~train_val_split, :, 9:]
    val_y = _label[~train_val_split, :, :]
    
    print(np.unique(np.argmax(val_y.reshape(-1, args.num_classes), axis=1), 
                    return_counts=True))
    
    
    print("train_data_index", train_data_index.shape)
    print("train_x", train_x.shape)
    print("train_y", train_y.shape)
    
    print("val_data_index", val_data_index.shape)
    print("val_x", val_x.shape)
    print("val_y", val_y.shape)
    
        
    ##########################################################################
    ## START THE CLASSIFICATION TASK

    # Define the callback and early stop
    if not os.path.exists("../models"):
        os.mkdir("../models")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath="../models/cvae_%0.f_%s.hdf5"%(args.epochs, timestr)
    ## Eraly stop
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]     
        
    # # Instantiate the model
    cvae = CVAE(args)   
    # Contruct the cave model    
    train =cvae.training() 
    train.summary() 
    
    # # Start training phase
    if args.train_mode:
        
        print("Start training the model...")           
        train.fit(x=[train_x, train_y],
                  y=train_y,
                  shuffle=True,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  verbose=1,
                  callbacks=callbacks_list,
                 validation_data=([val_x, val_y], val_y))
        train.load_weights(filepath)
        
    else:
        print('Run pretrained model')
        train.load_weights("../models/xxx.hdf5")
        
            
    # # Start inference phase
    x_encoder=cvae.X_encoder()
    decoder = cvae.Decoder()       
    x_encoder.summary()
    decoder.summary()
    
    x_latent = x_encoder.predict(val_x, batch_size=args.batch_size)
    y_primes = []
    for i, x_ in enumerate(x_latent):
        # sampling z from a normal distribution
        x_ = np.reshape(x_, [1, -1])
        z_sample = np.random.rand(1, args.z_dim)
        y_p = decoder.predict(np.column_stack([z_sample, x_]))
        y_primes.append(y_p)
    
    y_primes = np.reshape(y_primes, (-1, args.num_classes))
    
    ## Evaluation
    target_names = ['uc', 'tl', 'ps']
    eva = Evaluation(val_y.reshape(-1, args.num_classes), 
                     y_primes,
                     target_names)    
    confusion_matrix = eva.cf_matrix()    
    classification_report = eva.report()
    
    # Sum up the prediction for each trip and each junction
    
    





if __name__ == "__main__":
    main()
    
    
    
