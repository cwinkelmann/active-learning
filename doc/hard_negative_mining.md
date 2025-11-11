# Training with negatives


## Random Crop Augmentations
Kellenberger 2018 

Use the augmentation of ObjectAwareRandomCrop which returns a tile with at least one object in it with a probability p -1, or a an empty tile with a probablity of p.


## Hard Negative Mining
### Mining for Hard Negatives without correction

Delplanque et al. (2024) present a comprehensive study on the importance of hard negative mining in improving the performance of machine learning models, particularly in the context of image recognition and natural language processing tasks.


This is implemented here by using detections, take high confidente false positives, correct them using CVAT, and add them to the training set as negative examples.

This involves multiple steps:
1. getting the detections using a model
2. find the false positives
3. adding them to the training set as negative examples

```shell 
# produce a detections.csv
herdnet/tools/inference_test.py

human_in_the_loop/051_evaluate_point_detector.py

human_in_the_loop/091_HIT_simple_merge_hasty

```

### Mining for hard negatives by using the HITL process
```shell 
# create the correction jobs in CVAT
061_HIT_1_geospatial_batched_1

# download the corrected annotations
061_HIT_2_geospatial

# merge the annotations into the original annotations
090_HIT_merge_hasty

```
### Inferencing Detections
Use a config, which applies patched inference using the right evaluator config. 


## Mining for Hard Negatives with correction


This involves multiple steps:
1. getting the detections using a model
2. selecting the high confidence false positives
3. correcting them using CVAT
4. adding them to the training set as negative examples

```shell 
inference_test.py


human_in_the_loop/051_evaluate_point_detector.py
human_in_the_loop/060_HIT_fp.py
human_in_the_loop/061_HIT_2
human_in_the_loop/091_HIT_simple_merge_hasty

```