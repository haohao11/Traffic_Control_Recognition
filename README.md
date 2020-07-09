# Junctions
This is the project for detecting junction arm rules based on GPS signal data 


# Data process
This is the step to select the GPS trips (trajectories) in each junction.
Note that, for a long trip, it might pass multiple junctions by different distance. Only the trips that pass the given junction (meet the *lower_threshold*, e.g., 10m to the junction center) is included to that particular junction. The included trips are cutoff into *junc_trip* by a *upper_theshold* (e.g., 100m to the junction center). 
To distinguish the same trip id for multiple junctions, a unique id is assigned to the *junc_trip* across all the junctions. The *min_trips* is used to exclude the junctions that have trips less than the predefine threshold value.

**arm rules**
- **tr**: tram rails
- **uc**: uncontrolled sign
- **ys**: yield sign
- **ps** priority sign
- **sp**: stop sign
- **tl**: traffic light
- **ra**: roundabout
 
``` python
python scripts/trainer.py
```

The following figures demonstrate junctions with *lower_threshold*=10, *upper_theshold*=100, (m) *min_trips*=16.
The following table lists the trips in all the junctions.

| Item  | Value |
| ------------- | ------------- |
| Number of selected junctions  | 240  |
| Number of total remaining trips  | 15509  |
| Remaining arm rues  | tr, uc, ys, ps, tl |
| GPS points in each arm rule | 8590, 123820, 10867, 111683, 148735 |
| Trips in each arm rule | 384, 3950, 350, 5465, 5360 |

Firgure 1            |  Figure 2
:-------------------------:|:-------------------------:
![junction_4857](https://github.com/haohao11/Junctions/blob/master/analysis/junctTrajs_4857_139.png) |  ![arm_data_distribution](https://github.com/haohao11/Junctions/blob/master/analysis/data_distribution.png)

The *lower_threshold* and *min_trips* values are taken from the paper, but sometimes it is problomatic when the next junction is too close (see Figure 1) and many junsctions have less than 16 trips passed by, such as roundabout, stop and sign. The threshold values may need to be optimized by the classification experiments. The third figure shows the distributions of number of GPS points and number of trips in each arm rule.

# Some intermediate results

## Classification results for each sliding window 
sliding window = 8, stride=2
test/validation (70/30)
model: cvae_100_20200709-112451.hdf5

| CF matrix  | uc | tf  | ps |
| ------------- | ------------- | ------------- | ------------- |
| uc | **7722** | 767 | 1325 |
| tf | 1331 | **9305** | 2263 |
| ps | 1180 | 1116 | **7303** |

| Items  | precission | recall  | f1-score | support |
| ------------- | ------------- | ------------- | ------------- |------------- |
| uncontrolled (uc) | 0.75 | 0.79 | 0.77 | 9814 |
| traffic light (tf) | 0.83 | 0.72 | 0.77 | 12899 |
| priority sign (ps) | 0.67 | 0.76 | 0.71 | 9599 |
| accuracy |  |  | 0.75 | 32312 |
| macro avg | 0.75 | 0.76 | 0.75 | 32312 |
| weighted avg | 0.76 | 0.75 | 0.75 | 32312 |

## Classification results for each arm
test/validation (70/30)
| CF matrix  | uc | tf  | ps |
| ------------- | ------------- | ------------- | ------------- |
| uc | **130** | 1 | 8 |
| tf | 8 | **172** | 22 |
| ps | 11 | 11 | **190** |

| Items  | precission | recall  | f1-score | support |
| ------------- | ------------- | ------------- | ------------- |------------- |
| uncontrolled (uc) | 0.87 | 0.94 | 0.90 | 139 |
| traffic light (tf) | 0.93 | 0.85 | 0.89 | 202 |
| priority sign (ps) | 0.86 | 0.90 | 0.88 | 212 |
| accuracy |  |  | 0.89 | 553 |
| macro avg | 0.89 | 0.89 | 0.89 | 553 |
| weighted avg | 0.89 | 0.89 | 0.89 | 553 |


# ToDos
- write sliding window (Done).
- apply seq-to-seq classiers (CVAE model, Done).
- write evluation metrics (Done).
- Sum up the prediction for trips/junctions/arms.

# Questions
- Why are the speed provided by the files and speed calculated by data_utils not the same? Is the bias of transformation from lon/long to utm?
- How to partition the data, based on trips/junctions/sequence?


