# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:13:18 2020

@author: cheng
"""


import numpy as np
import folium
from folium.plugins import MarkerCluster


a = np.array([[1, 2, 3],
              [0, 2, 4]])

is_all_zero = np.any((a == 0))
print(is_all_zero)


map = folium.Map(location=[45.523, -122.675],
                       width=750, height=500)

marker_cluster = MarkerCluster().add_to(map)

map.save( '../test/map.html')