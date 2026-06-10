# check_labels.py

import pickle
from datasets.mosei_dataset import MOSEIDataset

with open("data/processed/aligned_50.pkl", "rb") as f:
    data = pickle.load(f)

dataset = MOSEIDataset(data, "train")

print("Samples after removing neutral:")
print(len(dataset))

print("First label:")
print(dataset[0]["label"])










# check_distribution.py

import pickle
from collections import Counter
from datasets.mosei_dataset import MOSEIDataset

with open("data/processed/aligned_50.pkl", "rb") as f:
    data = pickle.load(f)

dataset = MOSEIDataset(data, "train")

labels = []

for i in range(len(dataset)):
    labels.append(dataset[i]["label"].item())

print(Counter(labels))