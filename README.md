# Human In the Loop Data Analytics (HILDA) active learning methods to improve machine learning for counting small infrequent Objects in drone Images
This repository was developed around the Iguanas from Above Project to help with counting all the animals in on the Galapagos Islands. 
Some of this code was used for the thesis project [Automated Marine Iguana Detection Using Drone Imagery and Deep Learning on the Galápagos Islands](https://figshare.com/articles/thesis/Automated_Marine_Iguana_Detection_Using_Drone_Imagery_and_Deep_Learning_on_the_Gal_pagos_Islands/30719999?file=59865122)
It is a rough code collection which needs some refactoring.
It was built around HerdNet developed by Alexandre Delplanque https://github.com/Alexandre-Delplanque/HerdNet, which was forked to extent it and experiment with its function: https://github.com/cwinkelmann/HerdNet



## 🚀 Roadmap

### 📋 Development Tasks

- [ ] 🧹 **Clean Code Refactoring**  
  Improve code quality, remove redundancies, and enhance maintainability

- [ ] 🔄 **Continuous Integration**  
  Set up automated testing and deployment pipelines

- [ ] 🖥️ **GUI Development**  
  Create user-friendly graphical interface for easier interaction

- [ ] 🗺️ **QGIS Integration**  
  Enable seamless workflow integration with QGIS platform

- [ ] 🔌 **Decouple Inferencing**  
  Separate inference logic for improved modularity and scalability

## Setup

First checkout the Herdnet Repo alongside this.

```shell
# delete the old environment if it exists
conda env remove -n active_learning

# conda create -n active_learning python=3.11

conda env create -n active_learning -f environment.yml

# activate the environment
conda activate active_learning

# switch into the repo folder
cd ../HerdNet

conda env update -f environment.yml
```

### Overview of the code
Most is implemented using scripts. More accessabile entrypoints are due by end of the year 2025.

#### Create a database of images including distance between shots, speed, estimated gsd
```
post_flight/010_image_db.py 
```
Based on that image lebel database metrics for each flight can be calculated
```
post_flight/012_mission_metrics_overall.py
```

#### Convert Other Dataset Format to hasty.ai
For example: [Delplanque et al. (2022)](https://doi.org/10.58119/ULG/MIRUU5) other datasets can be convert to a unified format.
Hasty.ai was chosen because it can function rather as a database than just a annotation storage.
```
# convert
dataset_conversion/06_deplanque_general/060_delplanque_general.py
# look into box sizes, etc
dataset_conversion/06_deplanque_general/061_analyse_delplanque_general.py
```

#### Prepare a training dataset
Based on the previous step the script:
```
training_data_preparation/021_hasty_to_tile_point_detection.py
```
can convert the dataset to a train/val/test split crop images into tiles, overlap the tiles. Modify dataset_configs_delplanque accordingly.
This outputs Herdnet formatted point annotation CSV files and an image Folder.

#### Training
Now train a HerdNet model. Modify the configurations accordingly.
[HerdNet](https://github.com/Alexandre-Delplanque/HerdNet)

#### Geospatial Inference
extend your pythonpath with the path to your HerdNet implementation
PYTHONPATH=$PYTHONPATH:/home/cwinkelmann/work/Herdnet:/Users/christian/PycharmProjects/hnee/HerdNet
```
inferencing/012_herdnet_geospatial_inference_2.py
```

#### Look into the predictions
```
human_in_the_loop/051_evaluate_point_detector.py
```

#### Human Correction Upload
```
061_HIT_1_correct_predictions_1.py
```

#### Human Correction Download
```
061_HIT_geospatial_batched_2.py
```




# Details

## Data Preprocessing
Most of the images were taken with a DJI Mavic 2 Pro in JPG Format. At the first the data was curated in to systematic folder structure, then orthomosaics were created using DroneDeploy, Metashape, Pix4D Mapper. The annotations were created using hasty.ai and CVAT.


### Image Folder Structure
The data folder structure is organised around the islands of Galápagos, each island has its own folder with each flight in a subfolder.

From the drone images where copied from the SD card to their island/flight folder and renamed to a systematic naming scheme.

The naming scheme is as follows:
`<IslandShortCode>_<FullSiteCode>_DJI_<ImageNumber>_<Date>_<(Optional) DroneName>.JPG`



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



