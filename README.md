# Human In the Loop Data Analytics (HILDA) active learning methods to improve machine learning for counting small infrequent Objects in drone Images

This repository was developed around the Iguanas from Above Project to help with counting all the animals in on the Galapagsos Islands. 

## Setup
```shell
# delete the old environment if it exists
conda env remove -n carrot_active_learning

# conda create -n active_learning python=3.11

conda env create -n carrot_active_learning -f environment.yml

#conda install gdal==3.6.2

#pip install -r requirements.txt

# Now install the herdnet package
in the root of the herdnet package
conda env update -f environment.yml --dry-run
```


## Data Preprocessing
Most of the images were taken with a DJI Mavic 2 Pro in JPG Format. At the first the data was curated in to systematic folder structure, then orthomosaics were created using DroneDeploy, Metashape, Pix4D Mapper. The annotations were created using hasty.ai and CVAT.


### Image Folder Structure
The data folder structure is organised around the islands of Galápagos, each island has its own folder with each flight in a subfolder.

From the drone images where copied from the SD card to their island/flight folder and renamed to a systematic naming scheme.

The naming scheme is as follows:
`<IslandShortCode>_<FullSiteCode>_DJI_<ImageNumber>_<Date>_<(Optional) DroneName>.JPG`

The island mapping and island short codes are as follows:
```plaintext
"Genovesa": "Gen_G"
"Isabela": "Isa_IS"
"Marchena": "Mar_M"
"Pinta": "Pin_P"
"Pinzón": "Pnz_PZ"
"Rábida": "Rab_RA"
"San Cristóbal": "Scris_SR"
...
```


Where:
```plaintext
Iguanas_From_Above/
└── <year>/
    ├── <Island>/
    │   ├── <IslandCode>_<SiteCode><FlightNumber>_<DateIn DDMMYYYY Format>/
    │   │   ├── DJI_<Number>.JPG
```

using the script post_flight/001_find_unrenamed_image_folders those were renamed to the systematic naming scheme.
```plaintext
Iguanas_From_Above/
└── <year>/
    ├── <Island>/
    │   ├── <IslandCode><FullSiteCode><FlightNumber>_<DateIn DDMMYYYY Format>/
    │   │   ├── <IslandShortCode>_<FullSiteCode>_<FlightNumber>_<DateIn DDMMYYYY Format>.JPG
```

For instance:
```plaintext
├── Marchena/
│   ├── MNW02_07122021/
│   │   ├── DJI_0001.JPG

leads to
├── Marchena/
│   ├── Mar_MNW02_07122021/
│   │   ├── Mar_MNW02_DJI_0001_07122021.JPG
```


This leads to the following folder structure:
```plaintext
Iguanas_From_Above/
└── 2020_2021_2022_2023_2024/
    ├── Island_A/
    │   ├── BT01_11012023/
    │   │   ├── Island_A_BT01_DJI_0001_11012023.JPG
    │   │   ├── ...
    │   │   └── Island_A_BT01_DJI_0003_11012023.JPG
    │   ├── BT02_15012023/
    │   │   ├── Island_A_BT02_DJI_0001_15012023.JPG
    │   │   ├── ...
    │   │   └── Island_A_BT02_DJI_0002_15012023.JPG
    ├── Island_B/
    │   ├── CR01_05022023/
    │   │   ├── Island_B_CR01_DJI_0001_05022023.JPG
    │   │   └── Island_B_CR01_DJI_0002_05022023.JPG
    │   └── CR02_08022023/
    │       └── Island_B_CR02_DJI_0001_08022023.JPG
    └── Island_C/
        ├── FL01_28012023/
        │   ├── Island_C_FL01_DJI_0001_28012023.JPG
        │   ├── Island_C_FL01_DJI_0002_28012023.JPG
        │   └── Island_C_FL01_DJI_0003_28012023.JPG
        └── FL02_02022023/
            └── Island_C_FL02_DJI_0001_02022023.JPG

```

### Image Database Creation
To understand more about the data a database is created. Within the project 4 years of images were take, leading to 320.000 images, due to file corruption and an old MAVIC 1 315812 images were added to the database

```shell
post_flight/010_image_db.py # create the database
post_flight/011_convert_analysis_ready_metadata # convert to shapefile, csv, geopackage
```
Using the database one can inspect flights, Gimbal angles, shutter speeds easily. The output are a CSV, GeoParquet. 
Many columns are extracted from the EXIF and XMP data of the images, some are derived. 
Some like gps_longitude_ref can be found in active_learning/types/image_metadata.py
Most notable entries in there are
Drone position
- AbsoluteAltitude ( gps_altitude_ref 0 means above sea level, )
- RelativeAltitude
- FlightRollDegree
- FlightPitchDegree
- FlightYawDegree
- geometry ( when loades GeoDataFrame )

- island_code
- site_code
- flight_code
- YYYYMMDD ( date of the flight )

Derived Values from the above
Ground Sampling Distance (GSD) assumes a flat terrain and that the flight altitude is the same as the distance from drone to terrain altitude, which is usually not the case.
- gsd_abs_width_cm ( Ground Sampling Distance based on absolute Height in cm/pixel, refers to image center )
- gsd_rel_height_cm ( Ground Sampling Distance based on relative Height in cm/pixel, refers to image center )
- shift_mm ( how much the drone moved between two images in mm, based on speed and time difference )
- shift_pixels ( how much the drone moved between two images in pixels, based on speed and time difference and GSD )
- risk_score ( based on speed and shutter speed, how likely is motion blur )


- distance_to_prev ( Within the flight, the distance to the previous image in meters )
- time_diff_seconds ( time difference to the previous image in seconds )
- speed_m_per_s ( speed based on distance to previous and time difference in m/s )

```shell
post_flight/012_mission_metrics.py 
```


## Workflow of loading data, preparing the annotations and correct them the first time

### Single Images in Hasty.ai

TODO describe the database creation of that.

### Project Orthomosaic Counting
Internally iguanas were counted on DroneDeploy and Metashape orthomsaics in a single pass. The output were shapefiles with points where iguanas were located.
Due to manual processing filenames, projects etc. were not consistent. Therefore the shapefiles need to be reorganised into a systematic folder structure and naming scheme.

```shell
043_reorganise_shapefiles.py
```
### Preparation of third party reference data
All other datasets are going to be converted into the same format, which is a format used by hasty.ai.

See scripts/dataset_conversion for these
```shell
01_eikelboom
03_weinstein_birds
06_delplanque_general
07_AED # African Elephant Dataset
```

These will result in a hasty.json file and some reformated folders.

scripts/training_data_preparation



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