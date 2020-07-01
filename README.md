# Junctions
This is the project for detecting junction arm rules based on GPS signal data 


# Data process
This is the step to select the GPS trips (trajectories) in each junction.
Note that, for a long trip, it might pass multiple junctions by different distance. Only the trips that pass the given junction (meet the *lower_threshold*, e.g., 10m to the junction center) is included to that particular junction. The included trips are cutoff into *junc_trip* by a *upper_theshold* (e.g., 100m to the junction center). 
To distinguish the same trip id for multiple junctions, a unique id is assigned to the *junc_trip* across all the junctions. The *min_trips* is used to exclude the junctions that have trips less than the predefine threshold value.

**arm rules**
- **tr** tram rails
- **uc** uncontrolled sign
- **ys** yield sign
- **ps** priority sign
- **sp** stop sign
- **ra** roundabout
 
``` python
python scripts/read_junc.py
```

The following figure demonstrates a junction with *lower_threshold*=10, *upper_theshold*=100, (m) *min_trips*=16.
The *lower_threshold* and *min_trips* values are taken from the paper, but sometimes it is problomatic when the next junction is too close (see the second figure) and many junsctions have less than 16 trips passed by. The threshold values may need to be optimized by the classification experiments.
![junction_0089](https://github.com/haohao11/Junctions/blob/master/analysis/junctTrajs_0089_165.png)
![junction_4857](https://github.com/haohao11/Junctions/blob/master/analysis/junctTrajs_4857_139.png)

