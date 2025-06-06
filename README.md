# Human In the Loop Data Analytics (HILDA) active learning methods to improve machine learning for counting small infrequent Objects in drone Images

## Setup
```shell


conda create -n active_learning python=3.11

conda install gdal

pip install -r requirements.txt

```

## Workflow of loading data, preparing the annotations and correct them the first time


This repo

002_review_annotations.py

003_download_from_cvat.py


## Data exploration
Based on the hasty formatted dataset
### analyse the resulting Dataset
TODO
- check the distribution of the classes
- check the size of objects
- compare how difficult objects are distinguishable from the background
- check the distribution of the objects in the images




## Simple Training Workflow

### based on Hasty Annotations
Convert bigger images to smaller tiles, filter and convert annotations
```shell
001_hasty_to_tile.py
```
001_isaid_to_tile is the alternative for coco/isaid annotations

Assuming the annotations have a different format, i.e. Coco based iSAID convert them first to hasty.json





### Start the training

#### Basic Faster-RCNN
TODO 

#### Herdnet see this repo

#### YOLO

#### DeepFaune

#### deepforest


### Evaluation with a trained model
generate a prediction on the test set using herdnet
```shell
inference_test.py
```

This can be used too for inferencing on the whole dataset up to an orthomosaic.
An orthomosaic will be probably to big to be inferenced at once, therefore tiling it is necessary.
```shell
044_prepare_orthomosaics_classification.py 
```



then evaluate the results
```shell
051_evaluate_point_detector.py
```

generate predictions using SAHI with YOLO
```shell
050_evaluate_YOLO_detector.py
```


## Geospatial application of the trained models
Use a model and a geotiff to predict the objects in the geotiff 
```shell
052_shp2other.py 
```

## Training data creation
Asside from just trainig on existing data, how can new training data be created

### cut an orthomosaic into tiles
```shell
044_prepare_orthomosaics_classification.py

```

```shell
046_evaluate_orthomosaic_detection.py
```

### Simple Human In the Loop learning
These are at least two steps

#### upload images to CVAT
Based on bigger images
```shell
002_review_annotations.py
```

#### download them after correction
```shell
003_download_from_cvat.py
```

Based on tiles

```shell
060_HIT.py
061_HIT_2.py
```

TODO then reconstruct the original big images from the tiles and the annotations

## Build and Use an Image Database



# Bonus Topics
## Edge Blackout Augmentation
Since the objects are not very compact, an edge object can be cut off and is not detectable anymore. Blacking the annotation of the edge can be a good idea.


## Random empty sampling
Sample empty tiles in the dataload for each epoch to provide. When empty tiles are 100 times more abundant than occupied tiles just adding 10% of emptyis maybe not enough.  



## Homography based detection deduplication to avoid Orthomosaicing

### Stack based detection deduplication
use a single cropped detection thumbnail, template search it in any other image, analyse the image stack

### full overlap based detection deduplication
Do not count detections twice if they are fully contained in the overlap to another image

### automatic orthorectification using an orthomosaic
can it be done

## ERA5 Correlation of iguanas abundance to weather data
- ERA5 data for for climate
- Multispectral data for vegetation


## Second Stage Classification using a Vision Transformer
similar to deepfaunea, use a vision transformer to classify the detected objects or remove false positives


## Further Model exploration
Explore other models, like wildlifemapper, Florence LLM, CoundGD, 