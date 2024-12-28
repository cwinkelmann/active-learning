import json

from pathlib import Path

p = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/iSAID/train/Annotations/iSAID_train.json")


d = json.load(p.open("r"))

print(d.keys())