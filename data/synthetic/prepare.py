"""
Set vocabulary size V = 2048, sequence length S = 64 and vary only the number of points in a dataset.
"""

# src/generate_data.py
import os
import pickle

import numpy as np

V = 2048  # Vocabulary size
S = 64  # Sequence length
N = 2**16  # Number of sequences

train_ids = np.random.randint(0, V, (N, S)).astype(np.uint16)  # Uniformly sampled

print(f"Generated {N:,} sequences of length {S} with vocabulary size {V}.")
print(f"train_ids shape: {train_ids.shape}, dtype: {train_ids.dtype}")
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))

# Save meta info
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": V}, f)
