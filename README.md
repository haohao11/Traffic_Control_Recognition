# Junctions
Accurate information of traffic regulators at junctions is important for navigating and driving in cities. However, such information is often missing, incomplete or not up-to-date in digital maps due to the high cost, e.g., time and money, for data acquisition and updating. In this study we propose a crowdsourced method that harnesses the light-weight GPS tracks from commuting vehicles as Volunteered Geographic Information (VGI) for traffic regulator detection. We explore the novel idea of detecting traffic regulators by learning the movement patterns of vehicles at regulated locations. Vehicles' movement behavior was encoded in the form of speed-profiles, where both speed values and their sequential order during movement development were used as features in a three-class classification problem for the most common traffic regulators: traffic-lights, priority-signs and uncontrolled junctions. The method provides an average weighting function and a majority voting scheme to tolerate the errors in the VGI data. The sequence-to-sequence framework requires no extra overhead for data processing, which makes the method applicable for real-world traffic regulator detection tasks. The results showed that the deep-learning classifier Conditional Variational Autoencoder can predict regulators with 90% accuracy, outperforming a random forest classifier (88% accuracy) that uses the summarized statistics of movement as features. In our future work images and augmentation techniques can be leveraged to generalize the method's ability for classifying a greater variety of traffic regulator classes.


# Data process
This is the step to select the GPS trips (trajectories) in each junction.
Note that, for a long trip, it might pass multiple junctions by different distance. Only the trips that pass the given junction (meet the *lower_threshold*, e.g., 10m to the junction center) is included to that particular junction. The included trips are cutoff into *junc_trip* by a *upper_theshold* (e.g., 65m to the junction center). 
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

Due to the data protection, the raw data is not provided.
If you use the model for your own data, please cite the paper

``` html
@article{cheng2020traffic,
  title={Traffic Control Recognition with Speed-Profiles: A Deep Learning Approach},
  author={Cheng, Hao and Zourlidou, Stefania and Sester, Monika},
  journal={ISPRS International Journal of Geo-Information},
  year={2020},
}
```


The *lower_threshold* and *min_trips* values are taken from the paper, but sometimes it is problomatic when the next junction is too close (see Figure 1) and many junsctions have less than 16 trips passed by, such as roundabout, stop and sign. The threshold values may need to be optimized by the classification experiments. The third figure shows the distributions of number of GPS points and number of trips in each arm rule.


# The experiment (hyper)
```html
min_tripss = 16
upper_threshold = 65.5
lower_threshold = 10.0
window_size = 8
stride = 2
num_features = 7
num_classes = 3
encoder_dim = 128
z_dim = 2
z_decoder_dim = 128
hidden_size = 128
batch_size = 256
s_drop = 0.3
z_drop = 0.1
beta = 0.8
split = 0.7
lr = 1e-3
wpochs = 500
patience = 20
````

model: cvae_500_20201008-213111_03_01_90.hdf5.hdf5

| CF matrix  | uc | tf  | ps |
| ------------- | ------------- | ------------- | ------------- |
| uc | **3803** | 416 | 1109 |
| tf | 760 | **6013** | 1377 |
| ps | 516 | 742 | **3769** |

| Items  | precission | recall  | f1-score | support |
| ------------- | ------------- | ------------- | ------------- |------------- |
| uncontrolled (uc) | 0.75 | 0.71 | 0.73 | 5328 |
| traffic light (tf) | 0.84 | 0.74 | 0.78 | 8150 |
| priority sign (ps) | 0.60 | 0.75 | 0.67 | 5027 |
| accuracy |  |  | 0.73 | 18505 |
| macro avg | 0.73 | 0.73 | 0.73 | 18505 |
| weighted avg | 0.75 | 0.73 | 0.74 | 18505 |

## Classification results for each arm
test/validation (70/30)
| CF matrix  | uc | tf  | ps |
| ------------- | ------------- | ------------- | ------------- |
| uc | **120** | 3 | 15 |
| tf | 4 | **187** | 16 |
| ps | 12 | 7 | **196** |

| Items  | precission | recall  | f1-score | support |
| ------------- | ------------- | ------------- | ------------- |------------- |
| uncontrolled (uc) | 0.88 | 0.87 | 0.88 | 138 |
| traffic light (tf) | 0.95 | 0.90 | 0.93 | 202 |
| priority sign (ps) | 0.86 | 0.91 | 0.89 | 215 |
| accuracy |  |  | 0.90 | 560 |
| macro avg | 0.90 | 0.89 | 0.90 | 560 |
| weighted avg | 0.90 | 0.90 | 0.90 | 560 |
