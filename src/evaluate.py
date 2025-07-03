import math
import os

import numpy as np
import torch
from tqdm import tqdm

from model import GPT, GPTConfig

# -------------------- Configuration --------------------
data_dir = os.path.join("data", "synthetic")
train_bin_path = os.path.join(data_dir, "train.bin")

out_dir = "out/synthetic_7M_1M"
device = "cuda"
device_type = "cuda" if "cuda" in device else "cpu"
block_size = 64
vocab_size = 2048
batch_size = 2048  # evaluation micro-batch size

# -------------------- Load Model --------------------
ckpt_path = os.path.join(out_dir, "ckpt_synthetic_7M_1M.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint["model_args"]

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to(device)

# -------------------- Load Dataset Fully --------------------
data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
data = torch.from_numpy(np.array(data, dtype=np.int64))

N_total = data.size(0)
N_seq = (N_total - 1) // block_size  # number of (x,y) pairs
num_batches = math.ceil(N_seq / batch_size)

print(f"Evaluating {N_seq:,} sequences of length {block_size} in {num_batches} batches...")

# -------------------- Inference --------------------
total_log_likelihood = 0.0
with torch.no_grad():
    for i in tqdm(range(0, N_seq, batch_size), desc="Evaluating batches"):
        x_batch = []
        y_batch = []
        for j in range(i, min(i + batch_size, N_seq)):
            start = j * block_size
            x = data[start : start + block_size]
            y = data[start + 1 : start + 1 + block_size]
            x_batch.append(x)
            y_batch.append(y)
        X = torch.stack(x_batch).to(device)
        Y = torch.stack(y_batch).to(device)

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(X, Y)

            # Calculate log-likelihood for this batch
            # logits shape: (batch_size, seq_len, vocab_size)
            # Y shape: (batch_size, seq_len)
            batch_size_actual = Y.size(0)
            seq_len = Y.size(1)

            # Reshape for easier computation
            logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
            targets_flat = Y.view(-1)  # (batch_size * seq_len,)

            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)

            # Get log probabilities for the actual targets
            target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

            # Sum log probabilities (in natural log)
            batch_log_likelihood = target_log_probs.sum().item()

            total_log_likelihood += batch_log_likelihood

# -------------------- Memorization Computation --------------------
H_x = N_seq * block_size * math.log2(vocab_size)
H_K_x_given_theta = -total_log_likelihood / math.log(2)

# Memorization: mem(x, θ̂) = H(x) - H_K(x | θ̂)
memorization = H_x - H_K_x_given_theta
num_params = model.get_num_params()
alpha = memorization / num_params
print(f"Model's memorization capacity: {memorization:.2f} bits")
print(f"Alpha: {alpha:.2f} bits/param")
