# YOLO

```shell
# train boxes
yolo detect train data=/Users/christian/data/training_data/2025_01_11_segments/data_boxes.yaml model=yolo11n-seg.pt epochs=100 imgsz=512
````

```shell
# train segmentation masks

yolo detect train data=/Users/christian/data/training_data/2025_01_11_segments/data_boxes.yaml model=model=yolo11n-seg.pt epochs=10 imgsz=512

```


## Herdnet
```shell
Use the herdnet repo tools/train.py script
```