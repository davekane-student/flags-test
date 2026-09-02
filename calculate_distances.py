import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Find all flag image files
files = sorted(glob.glob("data/*.png"))
codes = [os.path.splitext(os.path.basename(f))[0] for f in files]

features = []
for f in files:
    with Image.open(f) as raw:
        # Resize to 32x32 RGB to capture spatial layout and shapes
        img = raw.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # 1. Spatial feature vector (32 x 32 x 3 = 3072 values)
        spatial_feat = arr.flatten()
        
        # 2. 3D Color histogram (8 x 8 x 8 bins = 512 values)
        hist, _ = np.histogramdd(
            arr.reshape(-1, 3),
            bins=(8, 8, 8),
            range=[(0, 1), (0, 1), (0, 1)]
        )
        hist_feat = (hist / hist.sum()).flatten()
        
        # Combine spatial structure (weight 0.7) and color distribution (weight 0.3)
        combined_feat = np.concatenate([
            spatial_feat * 0.7,
            hist_feat * 0.3 * np.sqrt(spatial_feat.size)
        ])
        features.append(combined_feat)

features = np.array(features)

# Compute pairwise Euclidean distance matrix
dist_matrix = cdist(features, features, metric="euclidean")

# Normalize distances to [0, 1] range
if dist_matrix.max() > 0:
    dist_matrix = dist_matrix / dist_matrix.max()

# Build long-format dataframe of distances
records = []
for i, code1 in enumerate(codes):
    for j, code2 in enumerate(codes):
        records.append({
            "code1": code1,
            "code2": code2,
            "distance": round(float(dist_matrix[i, j]), 4)
        })

df_dist = pd.DataFrame(records)
df_dist.to_csv("flag_distances.csv", index=False)
print(f"Saved {len(df_dist)} pairwise distances to flag_distances.csv")
