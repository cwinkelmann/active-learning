# Training data preparation

First step before training a model is to prepare the data in the right format. This includes downloading, converting and exploring the data.

The starting point is to have images and annotations. The annotations can be in different formats depending on the source of the data. Common formats include COCO, Pascal VOC, YOLO, and custom JSON formats.


The script **dataset_conversions/06_delplanque_general.py** shows how to convert the coco formatted annotations from Delplanque et al. (2024) to hasty.json format.



Then **scripts/training_data_preparation/021_hasty_to_tile_point_detection** illustrates how to convert the hasty.json annotations to tiled images and corresponding annotations for point detection tasks.

Then given the correctly formatted herdnet training config either on the crops, i.e. herdnet_format_512_160_crops.csv ( 512px patches with an overlap of 160px) or on the full images herdnet_format.csv, the training can be started using herdnet. The image folders are "crops_512_numNone_overlap160" or "Default" respectively.