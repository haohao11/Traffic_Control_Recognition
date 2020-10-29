# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:29:14 2020
This is the mudule to call:
    the read_junc module for data preprocess
    the data_utils module for loading data
    the model module for classification
    
data:
    0: id, (this is the unique id for the trips across the junctions)
    1: junc_id,
    2: junc_arm_rule,
    3: junc_arm,
    4: gid, 
    5: trip_id, 
    6: lat, 
    7: lon, 
    8: unixtime, 
    9: timestamp,
    10: junc_utm_to_center,
    11: utm_east, 
    12: utm_north, 
    13: utm_east_speed, 
    14: utm_east_speed, 
    15: speed_1, 
    16: speed_2, (old table speed) 
    17: delta_time
    18: angle
    
    
    
@author: cheng
"""

import argparse
import glob
import numpy as np
import os
import sys
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping
from junction_classier import Junction_class
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
    parser.add_argument('--upper_threshold', type=float, default=65.0, 
                        help='the upper bound distance [m] of each trip to the given junction')
    parser.add_argument('--lower_threshold', type=float, default=10.0, 
                        help='the lower bound distance [m] of each trip to the given junction')
    parser.add_argument('--process_data', type=bool, default=False, 
                        help='the lower bound distance [m] of each trip to the given junction')
    parser.add_argument('--window_size', type=int, default=8, 
                        help='sequence length for the sliding window')
    parser.add_argument('--stride', type=int, default=2, 
                        help='stride for the sliding window') 
    parser.add_argument('--num_features', type=int, default=6, 
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
                        help='The size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size')
    parser.add_argument('--s_drop', type=float, default=0.3, 
                        help='The dropout rate for trajectory sequence')
    parser.add_argument('--z_drop', type=float, default=0.1, 
                        help='The dropout rate for z input')
    parser.add_argument('--query_dim', type=int, default=64, 
                        help='The dimension of the query')
    parser.add_argument('--keyvalue_dim', type=int, default=64, 
                        help='The dimension for key and value')
    parser.add_argument('--beta', type=float, default=0.80, 
                        help='Loss weight')   
    parser.add_argument('--train_mode', type=bool, default=True, 
                        help='This is the training mode')
    parser.add_argument('--split', type=float, default=0.7, 
                        help='the split rate for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--aug_num', type=int, default=8, 
                        help='Number of augmentations')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=30, 
                        help='Maximum mumber of continuous epochs without converging')    
                          
    args = parser.parse_args(sys.argv[1:])
    return args


def normalization(data):
    '''
    MinMax normalization
    '''
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaler.data_max_
    normal_feature = scaler.transform(data)
    return normal_feature

def data_partition(data, args, data_summary=False):
    
    
    # split the data into training and testing
    feature_index = 0
    split = args.split
    # junc_arms = np.unique(data[:, 3])
    
    split_list = np.unique(data[:, feature_index])
    split_n = int(len(split_list)*split)
    np.random.seed(6) 
    index = np.random.choice(split_list.shape[0], split_n, replace=False) 
    train_ = split_list[index]
    # data = np.reshape(data, (-1, window_size, data.shape[-1]))
    train_val_split = []    
    for i in data[:, 0, feature_index]:
        train_val_split.append(i in train_)  
    print("train_val_split", np.unique(train_val_split, return_counts=True))
    # print(train_val_split)    
    # print("split_list", split_list)
    # sys.exit()
    
    # Summmary of the trips in each rule
    if data_summary:
        uc = [] ## 0
        tl = [] ## 1
        ps = [] ## 2
    
        data = data.reshape(-1, data.shape[-1])
        print("\nunique trips regarding junctions", len(np.unique(data[:, 0])))
        print("unique junctions", len(np.unique(data[:, 1])))
        print("overall trips", len(np.unique(data[:, 5])))
        print("unique junc_arm", len(np.unique(data[:, 3])))    
        print("unique junc_arm_rules", np.unique(data[:, 2]))
       
        for i in data:
            if i[2] == 0:
                if i[0] not in uc:
                    uc.append(i[0])
            elif i[2] == 1:
                if i[0] not in tl:
                    tl.append(i[0])
            elif i[2] == 2:
                if i[0] not in ps:
                    ps.append(i[0])                        
        print("uc: ", len(uc), "tl:", len(tl), "ps", len(ps))
    
    return np.asarray(train_val_split)
  

def main():
    
    args = parse_args()
    processeddata_dir = \
        "../Hanover_Dataset/HannoverDataset/processed_data/juncs_trips_%02.0f.npy"%args.min_trips
    rawdata_file = "../Hanover_Dataset/HannoverDataset"
    
    # Process and save the data
    if args.process_data:
        junctTrajs_dirs = sorted(glob.glob(
            os.path.join(rawdata_file, "junctTrajs/*.csv")))
        junctions_dir = os.path.join(rawdata_file, "junctions.csv")
           
        juncs_trips = preprocess(junctTrajs_dirs, junctions_dir, 
                                 min_trips=args.min_trips, 
                                 upper_threshold=args.upper_threshold, 
                                 lower_threshold=args.lower_threshold,
                                 window_size=args.window_size)        
        np.save(processeddata_dir, juncs_trips)
    else:
        juncs_trips = np.load(processeddata_dir)
    
    
    # Load the sequence data using the sliding window scheme
    data_loader = Load_data(juncs_trips, 
                            window_size=args.window_size,
                            stride=args.stride) 
        
    # Load the sequence data and get the data index
    data = [sequence for sequence in data_loader.sliding_window()]
    data = np.reshape(data, (-1, 19)) # data index + features = 10 + 9 = 19
    
    
    
    # Note, due to the data imbalance, 
    # merge tram_rails (-1) and yield_sigh (1) to priority_sign (2)
    if args.num_classes==3:
        data[data[:, 2]==-1, 2] = 2
        data[data[:, 2]==1, 2] = 2
        
        # # Filter out -1 and 1
        # data = data[data[:, 2]!=-1, :]
        # data = data[data[:, 2]!=1, :]
        
        # new target class: 
        # uncontrolled:0, 
        # traffic_light:1, 
        # tram_rails/yield_sigh/priority_sign
        data[data[:, 2]==4, 2] = 1
        
    # Filter out 3:"stop S." and 5:"roundabout"
    data = data[data[:, 2]!=3, :]
    data = data[data[:, 2]!=5, :]
        
    # Normalize the features
    data[:, 10:] = normalization(data[:, 10:])
    
    # Get the class label
    label = data[:, 2].astype(int)
    assert args.num_classes== len(np.unique(label)), "The number of classes is not correct"
    _label = np.eye(args.num_classes)[label].reshape(-1, args.window_size, args.num_classes)    
        
    # Question: how to do the data partitioning    
    data = np.reshape(data, (-1, args.window_size, 19))
    print(data.shape)
    
    train_val_split = data_partition(data, args)
    
    train_data_index = data[train_val_split, -1, :10] # the last step of the sliding window
    # 10/0: junc_utm_to_center,
    # 11/1: utm_east, 
    # 12/2: utm_north, 
    # 13/3: utm_east_speed, 
    # 14/4: utm_east_speed, 
    # 15/5: speed_1, 
    # 16/6: speed_2, (old table speed) 
    # 17/7: delta_time
    # 18/8: angle
    
  
# =============================================================================
#     train_x = data[train_val_split, :, 10:19]
#     train_x = np.concatenate((train_x[:, :, 0:6], train_x[:, :, 7:8]), axis=2) # skip the old speed
#     train_y = _label[train_val_split, -1, :] # the last step of the sliding window
#     
#     val_data_index = data[~train_val_split, -1, :10] # the last step of the sliding window
#     val_x = data[~train_val_split, :, 10:19]
#     val_x = np.concatenate((val_x[:, :, 0:6], val_x[:, :, 7:8]), axis=2)
#     val_y = _label[~train_val_split, -1, :] # the last step of the sliding window
# =============================================================================
    
    
    # ToDo, feature/ablation
    # remove delta_t
    train_x = data[train_val_split, :, 10:19]
    train_x = np.concatenate((train_x[:, :, 0:5], train_x[:, :, 7:8]), axis=2)
    # train_x = train_x[:, :, 2:5]
    train_y = _label[train_val_split, -1, :]   
    val_data_index = data[~train_val_split, -1, :10] 
    val_x = data[~train_val_split, :, 10:19]
    val_x = np.concatenate((val_x[:, :, 0:5], val_x[:, :, 7:8]), axis=2)
    # val_x = val_x[:, :, 2:5]
    val_y = _label[~train_val_split, -1, :] 
    
    
    
    
    
    
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
        # train.load_weights("../models/cvae_500_20201008-213111_03_01_90.hdf5")
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
        # train.load_weights("../models/cvae_200_20200709-211305.hdf5")
        train.load_weights("../models/cvae_500_20201008-213111_03_01_90.hdf5")
        
            
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
    print("Prediction for each sliding window...")
    target_names = ['uc', 'tl', 'ps']
    eva = Evaluation(val_y.reshape(-1, args.num_classes), 
                     y_primes,
                     target_names)    
    confusion_matrix = eva.cf_matrix()    
    classification_report = eva.report()
    
    # Sum up the prediction for each trip and each junction
    print("Prediction for each arm...")
    junc_classifier = Junction_class(val_data_index, val_y, y_primes)
    arm_gt, arm_pd = junc_classifier.avg_classfier()
    
    
    arm_eva = Evaluation(arm_gt, 
                     arm_pd[:, 0],
                     target_names,
                     arg=False)    
    arm_confusion_matrix = arm_eva.cf_matrix()    
    arm_classification_report = arm_eva.report()
    
    

    
    
    
    
    
    





if __name__ == "__main__":
    main()
    
    
    
