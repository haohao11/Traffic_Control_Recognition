# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:11:05 2020

@author: cheng
"""

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import utm


def preprocess(junctTrajs_dirs, junctions_dir, min_trips=16, upper_threshold=100, lower_threshold=10):
    '''
    Process the junction trip data
    Type: numpy
    Header:
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
    junctions = read_data(junctions_dir)
    
    juncs_trips = np.zeros((0, 12))    
    for junctTrajs_dir in junctTrajs_dirs:
        junc_id = int(junctTrajs_dir.split('\\')[-1].split('.')[0].split('_')[-1])
        junction = junctions.loc[junctions['junc_id']==junc_id]
        
        junc_data = read_data(junctTrajs_dir)
        junc_trips = read_trip(junc_data, junction, junc_id, 
                               min_trips=min_trips, 
                               upper_threshold=upper_threshold, 
                               lower_threshold=lower_threshold)           
        if junc_trips is not None:
            juncs_trips = np.vstack((juncs_trips, junc_trips))
            
        # if junc_id>=2000:
        #     break
                            
    # Add the unique id for each trip across the junctions  
    juncs_trips = add_index(juncs_trips)
        
    return juncs_trips
    

def read_data(dir):
    data = pd.read_csv(dir, sep=',')
    return data


def read_trip(junc_data, junction, junc_id, min_trips, upper_threshold, lower_threshold):
    '''
    This is the function to retrieve all the trips inside each junction
    Parameters
    ----------
    junc_data : pandas
        contains the trips information and arm rule inside each junction.
    junction : pandas
        provides the information of the junction location.
    junc_id : int
        the inction id.
        
    Return
    junc_trips : numpy array
    '''
    # Store the junction trip data
    junc_trips = np.zeros((0, 12))
    
    # Define the color of each rule
    colors = {-1:'k', 0:'g', 1:'r', 2:'c', 3:'m', 4:'y', 5:'b'}    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(-upper_threshold, upper_threshold)
    ax.set_ylim(-upper_threshold, upper_threshold)
        
    junc_utm = utm.from_latlon(junction.values[0, 1], junction.values[0, 2])
    junc_utm_east = junc_utm[0]
    junc_utm_north = junc_utm[1]    
    junc_data = junc_data.values
    
    count = 0
    for i in junc_data:
        trip_id = i[0]
        junc_arm_rule = i[-2]
                
        trip_dir = os.path.join("../Hanover_Dataset/HannoverDataset/trips", "trip_%s.csv"%str(trip_id))
        trip_data = read_data(trip_dir).values
        
        junc_utm = utm.from_latlon(trip_data[:, 2], trip_data[:, 3])
        
        xy = np.hstack((junc_utm[0].reshape(-1, 1), junc_utm[1].reshape(-1, 1))) \
            - [junc_utm_east, junc_utm_north]
        trip_data = np.concatenate((trip_data, xy), axis=1)
        trip_data = filter_space(trip_data, upper_threshold=upper_threshold, lower_threshold=lower_threshold)
        
        if trip_data is not None:
            trip_data = add_juncid(junc_id, junc_arm_rule, trip_data)
            junc_trips = np.vstack((junc_trips, trip_data))
            ax.plot(trip_data[:, -2], trip_data[:, -3], color=colors[junc_arm_rule])
            count += 1
            
    ax.plot([], [], color='k', label='tr') # tram rails
    ax.plot([], [], color='g', label='uc') # uncontrolled sign
    ax.plot([], [], color='r', label='ys') # yield sign
    ax.plot([], [], color='c', label='ps') # priority sign
    ax.plot([], [], color='m', label='sp') # stop sign
    ax.plot([], [], color='y', label='tl') # traffic light
    ax.plot([], [], color='b', label='ra') # roundabout
    
    plt.title('Junction %04.0f with %.0f tajectories'%(junc_id, len(junc_data)))
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    figures_dir = "../Hanover_Dataset/HannoverDataset/figures_for_juncs"
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    
    if count > 0:
        plt.savefig(os.path.join(figures_dir, 
                                  "junctTrajs_%04.0f_%.0f"%(junc_id, count)), dpi=200)
        plt.show()
    plt.gcf().clear()
    plt.close()
        
    print(junc_trips.shape, count)
    if count >= min_trips:
        return junc_trips
    
        
def filter_space(data, upper_threshold=100, lower_threshold=10):
    '''
    This the function to filter the trips regarding the corresonding intersection
    by a distance threshold to the center of the intersection
    '''
    distance = np.linalg.norm(data[:, -2:], axis=1).reshape(-1, 1)
    data = np.concatenate((data, distance), axis=1)
    data = data[data[:, -1]<=upper_threshold]

    # Here also need to check if the trip drives through the intersection
    if np.min(data[:, -1])<=lower_threshold:
        return data
    
    
def add_juncid(junc_id, junc_arm_rule, trip_data):
    junc_id_c = np.full((trip_data.shape[0], 1), junc_id)
    junc_arm_rule_c = np.full((trip_data.shape[0], 1), junc_arm_rule)
    trip_data = np.concatenate((junc_id_c, junc_arm_rule_c, trip_data), axis=1)
    return trip_data


def add_index(data):
    '''
    This is the function to get the statistics of the data

    '''
    junc_ids = np.unique(data[:, 0])
    print("\nNumber of selected junctions %.0f"%len(junc_ids))
    print("Number of remaining trips")
    
    # define the unique id for the trip across all the junctions
    id = 0
    
    arm_dic = {-1:"tram rails",
               0:"uncontrolled",
               1:"yield S.",
               2:"priority S.",
               3:"stop S.",
               4:"traffic L.",
               5:"roundabout"}
    
    _data = np.zeros((0, 13))
    _rows = np.zeros((0, 13))
    
    for junc_id in junc_ids:
        junc_data = data[data[:, 0]==junc_id, :]
        junc_trip_ids = np.unique(junc_data[:, 3])
        for junc_trip_id in junc_trip_ids:
            junc_trip = junc_data[junc_data[:, 3]==junc_trip_id, :]
            _id = np.full((junc_trip.shape[0], 1), id)
            junc_trip = np.concatenate((_id, junc_trip), axis=1)
            _data = np.vstack((_data, junc_trip))
            _rows = np.vstack((_rows, junc_trip[0:1, :]))
            id += 1
            
    total_trips = np.unique(_data[:, 0])
    uni_arms_points, counts_uni_arms_points = np.unique(_data[:, 2], return_counts=True)
    uni_arms_rows, counts_uni_arms_rows = np.unique(_rows[:, 2], return_counts=True)
        
    print("total_trips", len(total_trips))
    print("uni_arms", uni_arms_points)
    print("counts_uni_arms_points", counts_uni_arms_points)
    print("counts_uni_arms_rows", counts_uni_arms_rows)
        
    # plot the trips distribution
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    x = [i for i in range(len(uni_arms_points))]
    x_ticks = [arm_dic[i] for i in uni_arms_points]
    
    # ax1.grid()
    ax1.bar(x, counts_uni_arms_points)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title("Points in each arm rule")
    
    # ax2.grid()
    ax2.bar(x, counts_uni_arms_rows)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_ticks)
    ax2.tick_params(axis='x', rotation=45)    
    ax2.set_title("Trips in each arm rule")
    
    fig.tight_layout()
    plt.savefig("../analysis/data_distribution.png", dpi=300)
    plt.show()
    plt.gcf().clear()
    plt.close()
       
    return _data
    

