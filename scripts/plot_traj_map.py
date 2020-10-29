# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:48:21 2020

Thsi is the module to plot all the GPS tracks on the map

@author: cheng
"""

import folium
import numpy as np
import pandas as pd
import glob
import os
import utm

from folium.plugins import BeautifyIcon


def main():
    
    data_dir = '../Hanover_Dataset/HannoverDataset'
    junc_dir = os.path.join(data_dir, 'junctions.csv')
    trip_dirs = sorted(glob.glob(os.path.join(data_dir, 'trips', '*.csv')))
    
    junc_pos = read_junc(junc_dir)
    
    trips = []
    for trip_dir in trip_dirs:        
        trip = read_trip(trip_dir)
        trips.append(trip)
    
    plot_juncs(junc_pos, trips)
    
    







def read_junc(dir):
    junc_pos = pd.read_csv(dir, sep=',', header=0)
    junc_pos = junc_pos.values  
    return junc_pos[:, 1:]

def read_trip(dir):
    print(dir)
    trip = pd.read_csv(dir, sep=',', header=0)
    trip = trip.values
    return trip[:, 2:4]
    

def plot_juncs(junc_pos, trips):
    
    
    m = folium.Map(location=[52.3759, 9.7320],
                   zoom_start=13,
                   tiles='OpenStreetMap',
                   control_scale = True,
                   width=1000,
                   height = 1500)
    
   
        
        
    for trip in trips:
        folium.PolyLine(trip, color="blue", weight=2.5, opacity=1).add_to(m)
        
    for junc in junc_pos:        
       folium.CircleMarker(location=junc,
                           radius=1,
                           popup='Hannover',
                           color='crimson',
                           fill=True,
                           fill_color='crimson').add_to(m)
          
    vt = folium.FeatureGroup(name='vechile trajectories', col= "red")
    jc = folium.FeatureGroup(name='junctions', col= "red")
    m.add_child(jc)
    m.add_child(vt)
        
    m.save('index.html')


if __name__ == "__main__":
    main()


