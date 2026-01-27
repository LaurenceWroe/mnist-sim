#%%
"""
Compare models trained with different random seeds.
Track tensor similarity and covariance structure to understand
which digit relationships are stable vs seed-dependent.
"""
# import sys
# from pathlib import Path

# # Add project root to path
# project_root = Path(__file__).parent  # or Path.cwd() if running interactively
# sys.path.insert(0, str(project_root))

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

from functions.model import Model, Config, _collator
from functions.datasets import MNIST
from functions.tn_sim import get_interaction_matrix, get_interaction_tensor_with_embedding, int_tensor_similarity

device = "cpu"

#%%
# Create figures directory at the start
from pathlib import Path

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
print(f"Saving figures to: {figures_dir.absolute()}")


#%%
# Train models with different seeds
seeds = [42, 123, 456]
models = {}
histories = {}

# Load the MNIST dataset (directly on to a device for efficiency)
train_data = MNIST(train=True, download=True, device=device)
test_data = MNIST(train=False, download=True, device=device)

for seed in seeds:
    print(f"\n=== Training model with seed {seed} ===")
    
    model = Model.from_config(epochs=20, seed=seed).to(device)
    
    train, test = MNIST(train=True, device=device), MNIST(train=False, device=device)
    metrics = model.fit(train, test)
    
    models[seed] = model
    histories[seed] = metrics

    print(f"Seed {seed} - Final val acc: {metrics['val/acc'].iloc[-1]:.4f}")

#%%
# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for seed in seeds:
    hist = histories[seed]
    epochs = range(len(hist))
    
    axes[0].plot(epochs, hist['val/loss'], label=f'Seed {seed}', linewidth=2)
    axes[1].plot(epochs, hist['val/acc'], label=f'Seed {seed}', linewidth=2)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('Training Loss Across Seeds')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Training Accuracy Across Seeds')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "seed_training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining complete!")
print(f"Models trained: {list(models.keys())}")


#%%
# Compute interaction tensors for all models
print("\n=== Computing interaction tensors ===")

# Without embedding (hidden space)
B_tensors = {}
for seed in seeds:
    B = get_interaction_matrix(models[seed])
    B_tensors[seed] = B
    print(f"Seed {seed} - B shape: {B.shape}")

# With embedding (input space)
T_tensors = {}
for seed in seeds:
    T = get_interaction_tensor_with_embedding(models[seed])
    T_tensors[seed] = T
    print(f"Seed {seed} - T shape: {T.shape}")

#%%
# Compute pairwise similarities - Hidden space (B)
print("\n=== Pairwise Similarities - Hidden Space (B) ===")
print(f"{'Pair':<20} {'Similarity':<15}")
print("-" * 35)

for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i:]:
        sim = int_tensor_similarity(B_tensors[seed1], B_tensors[seed2])
        if seed1 == seed2:
            print(f"Seed {seed1} vs {seed2:<10} {sim:>14.6f}  (self)")
        else:
            print(f"Seed {seed1} vs {seed2:<10} {sim:>14.6f}")

#%%
# Compute pairwise similarities - Input space (T, with embedding)
print("\n=== Pairwise Similarities - Input Space (T, with embedding) ===")
print(f"{'Pair':<20} {'Similarity':<15}")
print("-" * 35)

for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i:]:
        sim = int_tensor_similarity(T_tensors[seed1], T_tensors[seed2])
        if seed1 == seed2:
            print(f"Seed {seed1} vs {seed2:<10} {sim:>14.6f}  (self)")
        else:
            print(f"Seed {seed1} vs {seed2:<10} {sim:>14.6f}")

# %%
