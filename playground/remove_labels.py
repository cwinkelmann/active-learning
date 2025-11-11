import pandas as pd


df_herdnet = pd.read_csv("/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/Delplanque2022_512_overlap_160_ebFalse/delplanque_train_with_hard_negatives/train/herdnet_format_512_160_crops.csv")
df_herdnet = df_herdnet[df_herdnet['species'] != 'hard_negative']

df_herdnet.to_csv("/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/Delplanque2022_512_overlap_160_ebFalse/delplanque_train_with_hard_negatives/train/herdnet_format_512_160_crops_hn_removed.csv", index=False)