import pickle

with open("data/processed/mosi_aligned_50.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(data["train"].keys())

print("Text :", data["train"]["text"][0].shape)
print("Audio:", data["train"]["audio"][0].shape)
print("Vision:", data["train"]["vision"][0].shape)

print("Classification Labels:")
print(data["train"]["classification_labels"][:5])

print("\nRegression Labels:")
print(data["train"]["regression_labels"][:5])