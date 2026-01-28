#%%
"""
Compare models trained with different random seeds.
Track tensor similarity and covariance structure to understand
which digit relationships are stable vs seed-dependent.
"""
import sys
from pathlib import Path
import os

os.chdir('/Users/wroe/Documents/AI/mnist-sim-clean')

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from functions.model import Model, Config
from functions.datasets import MNIST
from functions.tn_sim import get_interaction_matrix, tensor_similarity, model_similarity, tn_sim_martin

device = "cpu"

#%%
# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
print(f"Saving figures to: {figures_dir.absolute()}")

#%%
# Load the MNIST dataset
train_data = MNIST(train=True, download=True, device=device)
test_data = MNIST(train=False, download=True, device=device)

seeds = [42, 123, 456]
models = {}
histories = {}
all_checkpoints = {}

#%%
# Train models with checkpoints
for seed in seeds:
    print(f"\n=== Training model with seed {seed} ===")
    
    model = Model.from_config(epochs=20, seed=seed).to(device)
    
    # Train with checkpoints every 5 batches
    history, checkpoints = model.fit(
        train_data, 
        test_data, 
        record_every_n_batches=5, 
        save_checkpoints=True
    )
    
    models[seed] = model
    histories[seed] = history
    all_checkpoints[seed] = checkpoints
    
    print(f"Seed {seed} - Final val acc: {history['val/acc'].iloc[-1]:.4f}")
    print(f"Seed {seed} - Number of checkpoints: {len(checkpoints)}")

#%%
# Store final models
print("\n=== Final models ready ===")
final_models = {}
for seed in seeds:
    final_models[seed] = models[seed]
    print(f"Seed {seed} - Final model stored")

#%%
# Compute similarity evolution - both with and without embedding
print("\n=== Computing similarity evolution ===")

similarity_evolution = {
    'with_embedding': {},
    'without_embedding': {}
}

for seed1 in seeds:
    similarity_evolution['with_embedding'][seed1] = {}
    similarity_evolution['without_embedding'][seed1] = {}
    
    print(f"Processing seed {seed1}...")
    
    for seed2 in seeds:
        print(f"  Computing similarity to seed {seed2} final...")
        
        sims_with = []
        sims_without = []
        
        for cp in tqdm(all_checkpoints[seed1], desc=f"Seed {seed1} → {seed2}"):
            # Load checkpoint into temporary model
            model_temp = Model.from_config(epochs=20, seed=seed1).to(device)
            model_temp.load_state_dict(cp['state_dict'])
            
            # With embedding
            sim_with = model_similarity(
                model_temp, 
                final_models[seed2], 
                include_embedding=True, 
                symmetrize=True
            )
            sims_with.append(sim_with)
            
            # Without embedding
            # sim_without = model_similarity(
            #     model_temp, 
            #     final_models[seed2], 
            #     include_embedding=False, 
            #     symmetrize=True
            # )
            # sims_without.append(sim_without)

            # Without embedding using Martin's code
            sim_without = tn_sim_martin(
                model_temp, 
                final_models[seed2]
            )
            sims_without.append(sim_without)
        
        similarity_evolution['with_embedding'][seed1][seed2] = sims_with
        similarity_evolution['without_embedding'][seed1][seed2] = sims_without

#%%
# Plot 1: Training loss vs TN similarity to final (with and without embedding on same plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, seed in enumerate(seeds):
    batch_steps = [cp['batch'] for cp in all_checkpoints[seed]]
    val_losses = [histories[seed][histories[seed]['batch'] == batch]['val/loss'].values[0] 
                  for batch in batch_steps]
    val_acc= [histories[seed][histories[seed]['batch'] == batch]['val/acc'].values[0] 
                  for batch in batch_steps]
    
    ax = axes[idx]
    ax_twin = ax.twinx()
    
    # Loss on left axis
    # ax.plot(batch_steps, val_losses, 'b-', label='Val Loss', linewidth=2.5, alpha=0.8)
    ax.plot(batch_steps, val_acc, 'b-', label='Val Acc', linewidth=2.5, alpha=0.8)
    
    # Similarities on right axis
    ax_twin.plot(batch_steps, similarity_evolution['with_embedding'][seed][seed], 
                 'r-', label='TN Sim (with emb)', linewidth=2, alpha=0.8)
    
    ax_twin.plot(batch_steps, similarity_evolution['without_embedding'][seed][seed], 
                 'orange', linestyle='--', label='TN Sim (no emb)', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Batch Steps')
    ax.set_ylabel('Validation Loss', color='b')
    ax_twin.set_ylabel('TN Similarity to Final', color='r')
    ax.set_title(f'Seed {seed}')
    ax.tick_params(axis='y', labelcolor='b')
    ax_twin.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_twin.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig(figures_dir / "loss_vs_tn_similarity.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot 2: Training loss vs GRADIENT TN similarity to final (with and without embedding on same plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, seed in enumerate(seeds):
    batch_steps = [cp['batch'] for cp in all_checkpoints[seed]]
    val_losses = [histories[seed][histories[seed]['batch'] == batch]['val/loss'].values[0] 
                  for batch in batch_steps]
    val_acc= [histories[seed][histories[seed]['batch'] == batch]['val/acc'].values[0] 
                  for batch in batch_steps]
    
    ax = axes[idx]
    ax_twin = ax.twinx()
    
    # Loss on left axis
    ax.plot(batch_steps, val_losses, 'b-', label='Val Loss', linewidth=2.5, alpha=0.8)
    # ax.plot(batch_steps, val_acc, 'b-', label='Val Acc', linewidth=2.5, alpha=0.8)
    
    # Similarities on right axis
    dy_dx = np.gradient(similarity_evolution['with_embedding'][seed][seed], batch_steps)
    ax_twin.plot(batch_steps, dy_dx, 'r--', label='Gradient TN Sim (with emb)', linewidth=2, alpha=0.8)

    dy_dx = np.gradient(similarity_evolution['without_embedding'][seed][seed], batch_steps)
    ax_twin.plot(batch_steps, dy_dx, 'orange', linestyle='--', label='Gradient TN Sim (no emb)', linewidth=2, alpha=0.8)

    ax.set_xlabel('Batch Steps')
    ax.set_ylabel('Validation Loss', color='b')
    ax_twin.set_ylabel('Gradient TN Similarity to Final', color='r')
    ax.set_title(f'Seed {seed}')
    ax.tick_params(axis='y', labelcolor='b')
    ax_twin.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_twin.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig(figures_dir / "loss_vs_tn_similarity.png", dpi=300, bbox_inches='tight')
plt.show()


#%%
# Plot 3: Cross-seed similarity evolution (both embedding types on same plots)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, seed1 in enumerate(seeds):
    batch_steps = [cp['batch'] for cp in all_checkpoints[seed1]]
    
    for seed2 in seeds:
        if seed1 == seed2:
            # Self-similarity
            axes[idx].plot(batch_steps, similarity_evolution['with_embedding'][seed1][seed2],
                          label=f'{seed1} → itself (with emb)', linewidth=3, alpha=1.0, linestyle='-')
            axes[idx].plot(batch_steps, similarity_evolution['without_embedding'][seed1][seed2],
                          label=f'{seed1} → itself (no emb)', linewidth=3, alpha=0.7, linestyle='--')
        else:
            # Cross-seed similarity (only show with embedding to avoid clutter)
            axes[idx].plot(batch_steps, similarity_evolution['with_embedding'][seed1][seed2],
                          label=f'{seed1} → {seed2} (with emb)', linewidth=2, alpha=0.6, linestyle='-')
    
    axes[idx].set_xlabel('Batch Steps')
    axes[idx].set_ylabel('TN Similarity')
    axes[idx].set_title(f'Seed {seed1} Evolution')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(figures_dir / "cross_seed_similarity_combined.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot 4: Direct comparison of with vs without embedding for each seed
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, seed1 in enumerate(seeds):
    batch_steps = [cp['batch'] for cp in all_checkpoints[seed1]]
    
    for seed2 in seeds:
        # With embedding
        axes[idx].plot(batch_steps, similarity_evolution['with_embedding'][seed1][seed2],
                      label=f'→ {seed2} (with emb)', linewidth=2.5 if seed1==seed2 else 1.5, 
                      alpha=1.0 if seed1==seed2 else 0.6, linestyle='-')
        

        # Without embedding
        axes[idx].plot(batch_steps, similarity_evolution['without_embedding'][seed1][seed2],
                      label=f'→ {seed2} (no emb)', linewidth=2.5 if seed1==seed2 else 1.5, 
                      alpha=1.0 if seed1==seed2 else 0.6, linestyle='--')
    
    axes[idx].set_xlabel('Batch Steps')
    axes[idx].set_ylabel('TN Similarity')
    axes[idx].set_title(f'Seed {seed1}: With vs Without Embedding')
    axes[idx].legend(fontsize=7)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([-0.1, 1.05])

plt.tight_layout()
plt.savefig(figures_dir / "embedding_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
