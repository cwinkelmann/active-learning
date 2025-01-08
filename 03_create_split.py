"""
naive way of creating a datasplit


"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

p = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/training/2024-12-06/crops_1000")
p = Path("/Users/christian/data/training_data/2024_12_06/train/crops_512")
hl = "herdnet_format.csv"

df = pd.read_csv(p / hl)

# Perform a 70/30 split
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=False)

# Save the splits if needed
train_df.to_csv(p / "train_split.csv", index=False)
test_df.to_csv(p / "test_split.csv", index=False)

print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))