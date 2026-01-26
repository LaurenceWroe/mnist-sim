#%%
"""
Compare models trained with different random seeds.
Track tensor similarity and covariance structure to understand
which digit relationships are stable vs seed-dependent.
"""
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
from functions.tn_sim import tensor_similarity_single_layer, tensor_similarity_with_embedding, interaction_matrix_similarity_new
from functions.tn_covariance import interaction_output_covariance

device = "cpu"

#%%
# Create figures directory at the start
from pathlib import Path

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
print(f"Saving figures to: {figures_dir.absolute()}")

#%%
# Load data
print("Loading MNIST data...")
train_data = MNIST(train=True, download=True, device=device)
test_data = MNIST(train=False, download=True, device=device)

#%%
# Training function with dual checkpoint schedules
def train_model_dual_checkpoints(train_data, test_data, config, name):
    """
    Train with two checkpoint schedules:
    1. Dense: every 5 batches for first 20 epochs (for similarity plots)
    2. Sparse: 0, 100, 200, ..., 500, then every 250 (for covariance analysis)
    """
    model = Model(config)
    torch.manual_seed(config.seed)
    torch.set_grad_enabled(True)
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collator(None)
    )
    
    batches_per_epoch = len(loader)
    
    checkpoints_dense = []
    checkpoints_sparse = []
    
    history = {
        'batch_steps': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': [],
    }
    
    total_batch_count = 0
    dense_epoch_limit = 20
    
    print(f"\n=== Training {name} ===")
    pbar = tqdm(range(config.epochs), desc=name)
    
    for epoch in pbar:
        for batch_idx, (x, y) in enumerate(loader):
            loss, acc = model.train().step(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_batch_count += 1
            
            # Dense checkpoints
            if epoch < dense_epoch_limit and total_batch_count % 5 == 0:
                val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
                
                history['batch_steps'].append(total_batch_count)
                history['val_loss'].append(val_loss.item())
                history['val_acc'].append(val_acc.item())
                history['epochs'].append(epoch + batch_idx / batches_per_epoch)
                
                checkpoint = {
                    'batch_step': total_batch_count,
                    'epoch': epoch + batch_idx / batches_per_epoch,
                    'state_dict': deepcopy(model.state_dict()),
                }
                checkpoints_dense.append(checkpoint)
            
            # Sparse checkpoints
            save_sparse = False
            if total_batch_count <= 500 and total_batch_count % 100 == 0:
                save_sparse = True
            elif total_batch_count > 500 and (total_batch_count - 500) % 250 == 0:
                save_sparse = True
            
            if save_sparse:
                checkpoint_sparse = {
                    'batch_step': total_batch_count,
                    'epoch': epoch + batch_idx / batches_per_epoch,
                    'state_dict': deepcopy(model.state_dict()),
                }
                checkpoints_sparse.append(checkpoint_sparse)
        
        scheduler.step()
        
        # Continue tracking after epoch 20
        if epoch >= dense_epoch_limit and epoch % 5 == 0:
            val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
            
            history['batch_steps'].append(total_batch_count)
            history['val_loss'].append(val_loss.item())
            history['val_acc'].append(val_acc.item())
            history['epochs'].append(epoch)
            
            checkpoint = {
                'batch_step': total_batch_count,
                'epoch': epoch,
                'state_dict': deepcopy(model.state_dict()),
            }
            checkpoints_dense.append(checkpoint)
        
        if epoch % 10 == 0:
            val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
            pbar.set_postfix({'val_acc': f'{val_acc.item():.3f}'})
    
    # Save final
    val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
    
    final_checkpoint = {
        'batch_step': total_batch_count,
        'epoch': config.epochs,
        'state_dict': deepcopy(model.state_dict()),
    }
    
    checkpoints_dense.append(final_checkpoint)
    checkpoints_sparse.append(deepcopy(final_checkpoint))
    
    history['batch_steps'].append(total_batch_count)
    history['val_loss'].append(val_loss.item())
    history['val_acc'].append(val_acc.item())
    history['epochs'].append(config.epochs)
    
    torch.set_grad_enabled(False)
    return checkpoints_dense, checkpoints_sparse, history, model

#%%
# Train 3 models with different seeds
seeds = [42, 123, 456]
models_data = {}

for seed in seeds:
    config = Config(
        lr=1e-3,
        wd=0.5,
        epochs=100,
        batch_size=2048,
        d_hidden=256,
        n_layer=1,
        d_input=784,
        d_output=10,
        bias=False,
        residual=False,
        seed=seed,
    )
    
    checkpoints_dense, checkpoints_sparse, history, model = train_model_dual_checkpoints(
        train_data,
        test_data,
        config,
        f"Seed {seed}"
    )
    
    models_data[seed] = {
        'config': config,
        'checkpoints_dense': checkpoints_dense,
        'checkpoints_sparse': checkpoints_sparse,
        'history': history,
        'model': model,
    }

#%%
# Extract final models
print("\nExtracting final models...")
final_models = {}
for seed in seeds:
    model = Model(models_data[seed]['config'])
    model.load_state_dict(models_data[seed]['checkpoints_sparse'][-1]['state_dict'])
    final_models[seed] = model

#%%
# Compute pairwise tensor similarities between final models
print("\nComputing pairwise tensor similarities...")
similarities = {}
for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i:]:
        sim = interaction_matrix_similarity_new(final_models[seed1], final_models[seed2], layer_idx=0,include_embedding=True)
        similarities[(seed1, seed2)] = sim
        print(f"Seed {seed1} vs Seed {seed2}: {sim:.4f}")

#%%
# Compute cross-model similarities during training
print("\nComputing similarity evolution...")

# For each seed, compute similarity to all other final models
sims_evolution = {seed: {other_seed: [] for other_seed in seeds} for seed in seeds}

for seed in seeds:
    print(f"Processing seed {seed}...")
    for cp in tqdm(models_data[seed]['checkpoints_dense']):
        model_temp = Model(models_data[seed]['config'])
        model_temp.load_state_dict(cp['state_dict'])
        
        for other_seed in seeds:
            sim = interaction_matrix_similarity_new(model_temp, final_models[other_seed], layer_idx=0)
            sims_evolution[seed][other_seed].append(sim)

#%%
# Plot similarity evolution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, seed in enumerate(seeds):
    batch_steps = models_data[seed]['history']['batch_steps']
    
    for other_seed in seeds:
        if seed == other_seed:
            label = f'Seed {seed} → itself'
            linewidth = 3
            alpha = 1.0
        else:
            label = f'Seed {seed} → Seed {other_seed}'
            linewidth = 1.5
            alpha = 0.6
        
        axes[idx].plot(batch_steps, sims_evolution[seed][other_seed], 
                      label=label, linewidth=linewidth, alpha=alpha)
    
    axes[idx].set_xlabel('Batch Steps')
    axes[idx].set_ylabel('Tensor Similarity')
    axes[idx].set_title(f'Seed {seed} Evolution')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir /"seed_similarity_evolution.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for seed in seeds:
    batch_steps = models_data[seed]['history']['batch_steps']
    axes[0].plot(batch_steps, models_data[seed]['history']['val_loss'], 
                label=f'Seed {seed}', linewidth=2)
    axes[1].plot(batch_steps, models_data[seed]['history']['val_acc'], 
                label=f'Seed {seed}', linewidth=2)

axes[0].set_xlabel('Batch Steps')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('Training Loss Across Seeds')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Batch Steps')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Training Accuracy Across Seeds')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir /"seed_training_curves.png", dpi=300, bbox_inches='tight')
plt.show()



# #%%
# # Compute covariance matrices for all final models
# print("\nComputing covariance matrices...")
# covs = {}
# for seed in seeds:
#     cov = interaction_output_covariance(final_models[seed], final_models[seed], layer_idx=0).cpu().numpy()
#     covs[seed] = cov

# #%%
# # Visualize all three covariance matrices
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# # Row 1: With diagonal
# vmax_with = max([np.abs(covs[seed]).max() for seed in seeds])

# for idx, seed in enumerate(seeds):
#     im = axes[0, idx].imshow(covs[seed], cmap='RdBu_r', vmin=-vmax_with, vmax=vmax_with)
#     axes[0, idx].set_title(f'Seed {seed}\nWith Diagonal')
#     axes[0, idx].set_xlabel('Output Class')
#     axes[0, idx].set_ylabel('Output Class')
#     plt.colorbar(im, ax=axes[0, idx])

# # Row 2: Without diagonal
# covs_no_diag = {}
# for seed in seeds:
#     cov_viz = covs[seed].copy()
#     np.fill_diagonal(cov_viz, np.nan)
#     covs_no_diag[seed] = cov_viz

# vmax_without = max([np.nanmax(np.abs(covs_no_diag[seed])) for seed in seeds])

# for idx, seed in enumerate(seeds):
#     im = axes[1, idx].imshow(covs_no_diag[seed], cmap='RdBu_r', vmin=-vmax_without, vmax=vmax_without)
#     axes[1, idx].set_title(f'Seed {seed}\nWithout Diagonal')
#     axes[1, idx].set_xlabel('Output Class')
#     axes[1, idx].set_ylabel('Output Class')
#     plt.colorbar(im, ax=axes[1, idx])

# plt.tight_layout()
# plt.savefig(figures_dir /"seed_covariance_comparison.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Compute pairwise differences in covariance
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# pairs = [(seeds[0], seeds[1]), (seeds[0], seeds[2]), (seeds[1], seeds[2])]

# for idx, (seed1, seed2) in enumerate(pairs):
#     diff = covs[seed1] - covs[seed2]
#     vmax_diff = np.nanmax(np.abs(diff))
#     vmax_diff = 1.5
    
#     im = axes[idx].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
#     axes[idx].set_title(f'Difference: Seed {seed1} - Seed {seed2}')
#     axes[idx].set_xlabel('Output Class')
#     axes[idx].set_ylabel('Output Class')
#     plt.colorbar(im, ax=axes[idx])

# plt.tight_layout()
# plt.savefig(figures_dir /"seed_covariance_differences.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Analyze specific digit relationships across seeds
# print("\n=== Digit Pair Covariances Across Seeds ===")
# digit_pairs = [(0, 6), (1, 4), (3, 4), (0, 8), (1, 9), (3, 9), (6, 9)]

# print(f"{'Pair':<10} {'Seed 42':<10} {'Seed 123':<10} {'Seed 456':<10} {'Std Dev':<10}")
# print("-" * 60)

# for d1, d2 in digit_pairs:
#     values = [covs[seed][d1, d2] for seed in seeds]
#     std = np.std(values)
#     print(f"{d1}-{d2:<8} {values[0]:>9.4f} {values[1]:>9.4f} {values[2]:>9.4f} {std:>9.4f}")

# #%%
# # Identify stable vs variable relationships
# print("\n=== Stable vs Variable Digit Relationships ===")

# # Compute std dev for all off-diagonal pairs
# pair_stds = []
# for i in range(10):
#     for j in range(i+1, 10):
#         values = [covs[seed][i, j] for seed in seeds]
#         std = np.std(values)
#         pair_stds.append((i, j, std, np.mean(values)))

# pair_stds.sort(key=lambda x: x[2], reverse=True)

# print("\nMost variable relationships (high std across seeds):")
# for i, j, std, mean in pair_stds[:10]:
#     print(f"  Digits {i}-{j}: std={std:.4f}, mean={mean:.4f}")

# print("\nMost stable relationships (low std across seeds):")
# for i, j, std, mean in pair_stds[-10:]:
#     print(f"  Digits {i}-{j}: std={std:.4f}, mean={mean:.4f}")

# print("\nAnalysis complete!")



# #%%
# # Compute similarity between covariance matrices
# print("\n=== Covariance Matrix Similarity ===")

# def covariance_similarity(cov1, cov2):
#     """
#     Compute similarity between two covariance matrices.
#     Uses normalized Frobenius inner product (like tensor similarity).
#     """
#     # Flatten to vectors
#     cov1_flat = cov1.flatten()
#     cov2_flat = cov2.flatten()
    
#     # Compute cosine similarity
#     dot_product = np.dot(cov1_flat, cov2_flat)
#     norm1 = np.linalg.norm(cov1_flat)
#     norm2 = np.linalg.norm(cov2_flat)
    
#     similarity = dot_product / (norm1 * norm2)
#     return similarity

# # Compute all pairwise covariance similarities
# cov_similarities = {}
# for i, seed1 in enumerate(seeds):
#     for seed2 in seeds[i:]:
#         sim = covariance_similarity(covs[seed1], covs[seed2])
#         cov_similarities[(seed1, seed2)] = sim
#         print(f"Covariance similarity - Seed {seed1} vs Seed {seed2}: {sim:.4f}")

# #%%
# # Compare tensor similarity vs covariance similarity
# print("\n=== Comparison: Tensor Similarity vs Covariance Similarity ===")
# print(f"{'Pair':<20} {'Tensor Sim':<15} {'Covariance Sim':<15}")
# print("-" * 50)

# for (seed1, seed2) in similarities.keys():
#     if seed1 != seed2:  # Skip self-comparisons
#         tensor_sim = similarities[(seed1, seed2)]
#         cov_sim = cov_similarities[(seed1, seed2)]
#         print(f"Seed {seed1} vs {seed2:<10} {tensor_sim:>14.4f} {cov_sim:>14.4f}")

# #%%
# # Compute covariance similarity evolution during training
# print("\nComputing covariance similarity evolution...")

# # Get covariance at sparse checkpoints for each seed
# covs_evolution = {seed: [] for seed in seeds}

# for seed in seeds:
#     print(f"Computing covariances for seed {seed}...")
#     for cp in tqdm(models_data[seed]['checkpoints_sparse']):
#         model_temp = Model(models_data[seed]['config'])
#         model_temp.load_state_dict(cp['state_dict'])
        
#         cov = interaction_output_covariance(model_temp, model_temp, layer_idx=0).cpu().numpy()
#         covs_evolution[seed].append(cov)
        

# # Compute cross-seed covariance similarities over time
# cov_sims_evolution = {}
# for seed1 in seeds:
#     cov_sims_evolution[seed1] = {}
#     for seed2 in seeds:
#         print(f"Computing similarity evolution: Seed {seed1} vs Seed {seed2}")
#         sims = []
#         # Use minimum length in case they have different numbers of checkpoints
#         min_len = min(len(covs_evolution[seed1]), len(covs_evolution[seed2]))
#         for i in range(min_len):
#             sim = covariance_similarity(covs_evolution[seed1][i], covs_evolution[seed2][i])
#             sims.append(sim)
#         cov_sims_evolution[seed1][seed2] = sims

# #%%
# # Plot covariance similarity evolution
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for idx, seed in enumerate(seeds):
#     # Get batch steps from sparse checkpoints
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_sparse']]
    
#     for other_seed in seeds:
#         if seed == other_seed:
#             label = f'Seed {seed} → itself'
#             linewidth = 3
#             alpha = 1.0
#         else:
#             label = f'Seed {seed} → Seed {other_seed}'
#             linewidth = 2
#             alpha = 0.7
        
#         # Handle potential length mismatch
#         plot_len = len(cov_sims_evolution[seed][other_seed])
#         axes[idx].plot(batch_steps[:plot_len], cov_sims_evolution[seed][other_seed], 
#                       label=label, linewidth=linewidth, alpha=alpha)
    
#     axes[idx].set_xlabel('Batch Steps')
#     axes[idx].set_ylabel('Covariance Matrix Similarity')
#     axes[idx].set_title(f'Seed {seed} Covariance Evolution')
#     axes[idx].legend()
#     axes[idx].grid(True, alpha=0.3)
#     axes[idx].set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "seed_covariance_similarity_evolution.png", dpi=300, bbox_inches='tight')
# plt.show()



# #%%
# # Compute covariance similarity evolution during training
# # Compare each checkpoint to its own final model
# print("\nComputing covariance similarity to final model...")

# # Get final covariance for each seed
# final_covs = {}
# for seed in seeds:
#     final_model = Model(models_data[seed]['config'])
#     final_model.load_state_dict(models_data[seed]['checkpoints_sparse'][-1]['state_dict'])
#     final_cov = interaction_output_covariance(final_model, final_model, layer_idx=0).cpu().numpy()
#     final_covs[seed] = final_cov

# # Compute similarity to final for each checkpoint
# cov_sims_to_final = {seed: [] for seed in seeds}

# for seed in seeds:
#     print(f"Computing covariance evolution for seed {seed}...")
#     for cp in tqdm(models_data[seed]['checkpoints_sparse']):
#         model_temp = Model(models_data[seed]['config'])
#         model_temp.load_state_dict(cp['state_dict'])
        
#         cov_temp = interaction_output_covariance(model_temp, model_temp, layer_idx=0).cpu().numpy()
#         sim = covariance_similarity(cov_temp, final_covs[seed])
#         cov_sims_to_final[seed].append(sim)

# #%%
# # Plot: Tensor similarity vs Covariance similarity (both to final model)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for idx, seed in enumerate(seeds):
#     batch_steps_dense = models_data[seed]['history']['batch_steps']
#     batch_steps_sparse = [cp['batch_step'] for cp in models_data[seed]['checkpoints_sparse']]
    
#     # Tensor similarity (from before)
#     axes[idx].plot(batch_steps_dense, sims_evolution[seed][seed], 
#                    label='Tensor Similarity', linewidth=2, alpha=0.7, color='blue')
    
#     # Covariance similarity
#     axes[idx].plot(batch_steps_sparse, cov_sims_to_final[seed], 
#                    label='Covariance Similarity', linewidth=2, alpha=0.7, color='orange')
    
#     axes[idx].set_xlabel('Batch Steps')
#     axes[idx].set_ylabel('Similarity to Final Model')
#     axes[idx].set_title(f'Seed {seed}: Convergence Comparison')
#     axes[idx].legend()
#     axes[idx].grid(True, alpha=0.3)
#     axes[idx].set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "tensor_vs_covariance_convergence.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Combined plot: All seeds, both metrics
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Left: Tensor similarity to final
# for seed in seeds:
#     batch_steps = models_data[seed]['history']['batch_steps']
#     axes[0].plot(batch_steps, sims_evolution[seed][seed], 
#                 label=f'Seed {seed}', linewidth=2)

# axes[0].set_xlabel('Batch Steps')
# axes[0].set_ylabel('Tensor Similarity to Final')
# axes[0].set_title('Tensor Similarity Convergence')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
# axes[0].set_ylim([0, 1.05])

# # Right: Covariance similarity to final
# for seed in seeds:
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_sparse']]
#     axes[1].plot(batch_steps, cov_sims_to_final[seed], 
#                 label=f'Seed {seed}', linewidth=2, marker='o', markersize=3)

# axes[1].set_xlabel('Batch Steps')
# axes[1].set_ylabel('Covariance Similarity to Final')
# axes[1].set_title('Covariance Similarity Convergence')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)
# axes[1].set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "all_seeds_convergence_comparison.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Print convergence statistics
# print("\n=== Convergence Statistics ===")
# print(f"{'Seed':<10} {'Tensor (batch 500)':<20} {'Covariance (batch 500)':<20}")
# print("-" * 60)

# for seed in seeds:
#     # Find similarity at batch 500 (or closest)
#     dense_idx = min(range(len(models_data[seed]['history']['batch_steps'])), 
#                     key=lambda i: abs(models_data[seed]['history']['batch_steps'][i] - 500))
#     sparse_idx = min(range(len(models_data[seed]['checkpoints_sparse'])), 
#                      key=lambda i: abs(models_data[seed]['checkpoints_sparse'][i]['batch_step'] - 500))
    
#     tensor_sim_500 = sims_evolution[seed][seed][dense_idx]
#     cov_sim_500 = cov_sims_to_final[seed][sparse_idx]
    
#     print(f"Seed {seed:<6} {tensor_sim_500:>19.4f} {cov_sim_500:>19.4f}")

# print("\nKey insight: Covariance converges much faster than raw tensors!")


# #%%
# # Compute separate similarities for diagonal and off-diagonal
# print("\n=== Diagonal vs Off-diagonal Similarity Analysis ===")

# def diagonal_similarity(cov1, cov2):
#     """Similarity of just diagonal elements (10 values)"""
#     diag1 = np.diag(cov1)
#     diag2 = np.diag(cov2)
    
#     dot = np.dot(diag1, diag2)
#     norm1 = np.linalg.norm(diag1)
#     norm2 = np.linalg.norm(diag2)
    
#     return dot / (norm1 * norm2)

# def offdiagonal_similarity(cov1, cov2):
#     """Similarity of just off-diagonal elements (90 values)"""
#     # Extract off-diagonal
#     mask = ~np.eye(10, dtype=bool)
#     off_diag1 = cov1[mask]
#     off_diag2 = cov2[mask]
    
#     dot = np.dot(off_diag1, off_diag2)
#     norm1 = np.linalg.norm(off_diag1)
#     norm2 = np.linalg.norm(off_diag2)
    
#     # Handle case where norm is zero
#     if norm1 == 0 or norm2 == 0:
#         return 0.0
    
#     return dot / (norm1 * norm2)

# # Plot diagonal vs off-diagonal similarity evolution (DENSE)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for idx, seed in enumerate(seeds):
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_dense']]
    
#     # Use the DENSE versions computed above
#     axes[idx].plot(batch_steps, diag_sims_to_final_dense[seed], 
#                    label='Diagonal Only', linewidth=2, alpha=0.8, linestyle='--')
#     axes[idx].plot(batch_steps, offdiag_sims_to_final_dense[seed], 
#                    label='Off-diagonal Only', linewidth=2, alpha=0.8, linestyle=':')
    
#     axes[idx].set_xlabel('Batch Steps')
#     axes[idx].set_ylabel('Similarity to Final')
#     axes[idx].set_title(f'Seed {seed}: Component Breakdown')
#     axes[idx].legend()
#     axes[idx].grid(True, alpha=0.3)
#     axes[idx].set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "diagonal_vs_offdiagonal_similarity_dense.png", dpi=300, bbox_inches='tight')
# plt.show()

# # Also compute full covariance similarity on dense checkpoints
# cov_sims_to_final_dense = {seed: [] for seed in seeds}

# for seed in seeds:
#     final_cov = final_covs[seed]
#     for cp in tqdm(models_data[seed]['checkpoints_dense'], desc=f"Seed {seed}"):
#         model_temp = Model(models_data[seed]['config'])
#         model_temp.load_state_dict(cp['state_dict'])
#         cov_temp = interaction_output_covariance(model_temp, model_temp, layer_idx=0).cpu().numpy()
#         sim = covariance_similarity(cov_temp, final_cov)
#         cov_sims_to_final_dense[seed].append(sim)

# # Then plot all three
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for idx, seed in enumerate(seeds):
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_dense']]
    
#     axes[idx].plot(batch_steps, cov_sims_to_final_dense[seed], 
#                    label='Full Covariance', linewidth=2.5, alpha=0.8)
#     axes[idx].plot(batch_steps, diag_sims_to_final_dense[seed], 
#                    label='Diagonal Only', linewidth=2, alpha=0.8, linestyle='--')
#     axes[idx].plot(batch_steps, offdiag_sims_to_final_dense[seed], 
#                    label='Off-diagonal Only', linewidth=2, alpha=0.8, linestyle=':')
    
#     axes[idx].set_xlabel('Batch Steps')
#     axes[idx].set_ylabel('Similarity to Final')
#     axes[idx].set_title(f'Seed {seed}: Component Breakdown')
#     axes[idx].legend()
#     axes[idx].grid(True, alpha=0.3)
#     axes[idx].set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "diagonal_vs_offdiagonal_similarity_dense.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Test on initial covariances
# print("\nInitial covariances (batch 0):")
# print(f"{'Pair':<20} {'Full':<10} {'Diagonal':<10} {'Off-diag':<10}")
# print("-" * 50)

# for seed1 in seeds:
#     for seed2 in seeds:
#         if seed1 < seed2:
#             sim_full = covariance_similarity(init_covs[seed1], init_covs[seed2])
#             sim_diag = diagonal_similarity(init_covs[seed1], init_covs[seed2])
#             sim_off = offdiagonal_similarity(init_covs[seed1], init_covs[seed2])
            
#             print(f"Seed {seed1} vs {seed2:<9} {sim_full:>9.4f} {sim_diag:>9.4f} {sim_off:>9.4f}")

# #%%
# # Compute evolution of diagonal vs off-diagonal similarity
# print("\nComputing diagonal and off-diagonal similarity evolution...")

# diag_sims_to_final = {seed: [] for seed in seeds}
# offdiag_sims_to_final = {seed: [] for seed in seeds}

# for seed in seeds:
#     print(f"Processing seed {seed}...")
#     final_cov = final_covs[seed]
    
#     for cp in tqdm(models_data[seed]['checkpoints_dense']):
#         model_temp = Model(models_data[seed]['config'])
#         model_temp.load_state_dict(cp['state_dict'])
        
#         cov_temp = interaction_output_covariance(model_temp, model_temp, layer_idx=0).cpu().numpy()
        
#         diag_sim = diagonal_similarity(cov_temp, final_cov)
#         offdiag_sim = offdiagonal_similarity(cov_temp, final_cov)
        
#         diag_sims_to_final[seed].append(diag_sim)
#         offdiag_sims_to_final[seed].append(offdiag_sim)

# #%%
# # Plot diagonal vs off-diagonal similarity evolution
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for idx, seed in enumerate(seeds):
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_dense']]
    
#     axes[idx].plot(batch_steps, cov_sims_to_final[seed], 
#                    label='Full Covariance', linewidth=2.5, alpha=0.8)
#     axes[idx].plot(batch_steps, diag_sims_to_final[seed], 
#                    label='Diagonal Only', linewidth=2, alpha=0.8, linestyle='--')
#     axes[idx].plot(batch_steps, offdiag_sims_to_final[seed], 
#                    label='Off-diagonal Only', linewidth=2, alpha=0.8, linestyle=':')
    
#     axes[idx].set_xlabel('Batch Steps')
#     axes[idx].set_ylabel('Similarity to Final')
#     axes[idx].set_title(f'Seed {seed}: Component Breakdown')
#     axes[idx].legend()
#     axes[idx].grid(True, alpha=0.3)
#     axes[idx].set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "diagonal_vs_offdiagonal_similarity.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Compare across seeds - off-diagonal only
# print("\n=== Cross-seed Off-diagonal Similarity ===")

# # Compute off-diagonal similarity between different seeds at final
# print("\nFinal models (off-diagonal only):")
# for seed1 in seeds:
#     for seed2 in seeds:
#         if seed1 < seed2:
#             sim_off = offdiagonal_similarity(final_covs[seed1], final_covs[seed2])
#             print(f"Seed {seed1} vs {seed2}: {sim_off:.4f}")

# #%%
# # Plot: All seeds, off-diagonal similarity only
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# for seed in seeds:
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_dense']]
#     ax.plot(batch_steps, offdiag_sims_to_final[seed], 
#             label=f'Seed {seed}', linewidth=2.5, marker='o', markersize=4)

# ax.set_xlabel('Batch Steps')
# ax.set_ylabel('Off-diagonal Similarity to Final')
# ax.set_title('Digit Relationships Convergence\n(Off-diagonal elements only)')
# ax.legend()
# ax.grid(True, alpha=0.3)
# ax.set_ylim([0, 1.05])

# plt.tight_layout()
# plt.savefig(figures_dir / "offdiagonal_similarity_convergence.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Summary statistics
# print("\n=== Summary: What Converges When? ===")

# for seed in seeds:
#     # Find when each metric crosses 0.99
#     batch_steps = [cp['batch_step'] for cp in models_data[seed]['checkpoints_dense']]
    
#     diag_99 = next((bs for bs, sim in zip(batch_steps, diag_sims_to_final[seed]) if sim > 0.99), "Never")
#     offdiag_99 = next((bs for bs, sim in zip(batch_steps, offdiag_sims_to_final[seed]) if sim > 0.99), "Never")
#     full_99 = next((bs for bs, sim in zip(batch_steps, cov_sims_to_final[seed]) if sim > 0.99), "Never")
    
#     print(f"\nSeed {seed} - Batch steps to reach 99% similarity:")
#     print(f"  Diagonal:     {diag_99}")
#     print(f"  Off-diagonal: {offdiag_99}")
#     print(f"  Full:         {full_99}")



# # #%%
# # # Modify the training function to include batch step 0 and very early checkpoints
# # def train_model_dual_checkpoints_with_init(train_data, test_data, config, name):
# #     """
# #     Train with checkpoints including initialization (batch 0).
# #     Dense: batch 0, then every 5 for first 20 epochs
# #     Sparse: batch 0, 10, 20, 50, 100, 200, ..., 500, then every 250
# #     """
# #     model = Model(config)
# #     torch.manual_seed(config.seed)
# #     torch.set_grad_enabled(True)
    
# #     optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
# #     scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
# #     loader = DataLoader(
# #         train_data,
# #         batch_size=config.batch_size,
# #         shuffle=True,
# #         drop_last=True,
# #         collate_fn=_collator(None)
# #     )
    
# #     batches_per_epoch = len(loader)
    
# #     checkpoints_dense = []
# #     checkpoints_sparse = []
    
# #     history = {
# #         'batch_steps': [],
# #         'val_loss': [],
# #         'val_acc': [],
# #         'epochs': [],
# #     }
    
# #     # SAVE INITIALIZATION (batch 0)
# #     model.eval()
# #     with torch.no_grad():
# #         val_loss, val_acc = model.step(test_data.x, test_data.y)
    
# #     init_checkpoint = {
# #         'batch_step': 0,
# #         'epoch': 0,
# #         'state_dict': deepcopy(model.state_dict()),
# #     }
# #     checkpoints_dense.append(init_checkpoint)
# #     checkpoints_sparse.append(deepcopy(init_checkpoint))
    
# #     history['batch_steps'].append(0)
# #     history['val_loss'].append(val_loss.item())
# #     history['val_acc'].append(val_acc.item())
# #     history['epochs'].append(0)
    
# #     total_batch_count = 0
# #     dense_epoch_limit = 20
    
# #     print(f"\n=== Training {name} ===")
# #     pbar = tqdm(range(config.epochs), desc=name)
    
# #     for epoch in pbar:
# #         for batch_idx, (x, y) in enumerate(loader):
# #             loss, acc = model.train().step(x, y)
            
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
            
# #             total_batch_count += 1
            
# #             # Dense checkpoints: every 5 batches for first 20 epochs
# #             if epoch < dense_epoch_limit and total_batch_count % 5 == 0:
# #                 val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
                
# #                 history['batch_steps'].append(total_batch_count)
# #                 history['val_loss'].append(val_loss.item())
# #                 history['val_acc'].append(val_acc.item())
# #                 history['epochs'].append(epoch + batch_idx / batches_per_epoch)
                
# #                 checkpoint = {
# #                     'batch_step': total_batch_count,
# #                     'epoch': epoch + batch_idx / batches_per_epoch,
# #                     'state_dict': deepcopy(model.state_dict()),
# #                 }
# #                 checkpoints_dense.append(checkpoint)
            
# #             # Sparse checkpoints: very dense at start, then sparser
# #             save_sparse = False
# #             if total_batch_count in [10, 20, 50]:  # Very early
# #                 save_sparse = True
# #             elif total_batch_count <= 500 and total_batch_count % 100 == 0:
# #                 save_sparse = True
# #             elif total_batch_count > 500 and (total_batch_count - 500) % 250 == 0:
# #                 save_sparse = True
            
# #             if save_sparse:
# #                 checkpoint_sparse = {
# #                     'batch_step': total_batch_count,
# #                     'epoch': epoch + batch_idx / batches_per_epoch,
# #                     'state_dict': deepcopy(model.state_dict()),
# #                 }
# #                 checkpoints_sparse.append(checkpoint_sparse)
        
# #         scheduler.step()
        
# #         # Continue tracking after epoch 20
# #         if epoch >= dense_epoch_limit and epoch % 5 == 0:
# #             val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
            
# #             history['batch_steps'].append(total_batch_count)
# #             history['val_loss'].append(val_loss.item())
# #             history['val_acc'].append(val_acc.item())
# #             history['epochs'].append(epoch)
            
# #             checkpoint = {
# #                 'batch_step': total_batch_count,
# #                 'epoch': epoch,
# #                 'state_dict': deepcopy(model.state_dict()),
# #             }
# #             checkpoints_dense.append(checkpoint)
        
# #         if epoch % 10 == 0:
# #             val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
# #             pbar.set_postfix({'val_acc': f'{val_acc.item():.3f}'})
    
# #     # Save final
# #     val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
    
# #     final_checkpoint = {
# #         'batch_step': total_batch_count,
# #         'epoch': config.epochs,
# #         'state_dict': deepcopy(model.state_dict()),
# #     }
    
# #     checkpoints_dense.append(final_checkpoint)
# #     checkpoints_sparse.append(deepcopy(final_checkpoint))
    
# #     history['batch_steps'].append(total_batch_count)
# #     history['val_loss'].append(val_loss.item())
# #     history['val_acc'].append(val_acc.item())
# #     history['epochs'].append(config.epochs)
    
# #     torch.set_grad_enabled(False)
# #     return checkpoints_dense, checkpoints_sparse, history, model

# # #%%
# # # Retrain the models with initialization checkpoints
# # seeds = [42, 123, 456]
# # models_data = {}

# # for seed in seeds:
# #     config = Config(
# #         lr=1e-3,
# #         wd=0.5,
# #         epochs=100,
# #         batch_size=2048,
# #         d_hidden=256,
# #         n_layer=1,
# #         d_input=784,
# #         d_output=10,
# #         bias=False,
# #         residual=False,
# #         seed=seed,
# #     )
    
# #     checkpoints_dense, checkpoints_sparse, history, model = train_model_dual_checkpoints_with_init(
# #         train_data,
# #         test_data,
# #         config,
# #         f"Seed {seed}"
# #     )
    
# #     models_data[seed] = {
# #         'config': config,
# #         'checkpoints_dense': checkpoints_dense,
# #         'checkpoints_sparse': checkpoints_sparse,
# #         'history': history,
# #         'model': model,
# #     }
# # # %%





# # #%%
# # # Compute cross-model covariance matrices
# # print("\n=== Cross-Model Covariance Matrices ===")

# # # Compute all pairwise cross-covariances
# # cross_covs = {}

# # for i, seed1 in enumerate(seeds):
# #     for seed2 in seeds:
# #         if seed1 != seed2:
# #             print(f"Computing cross-covariance: Seed {seed1} vs Seed {seed2}")
# #             cross_cov = interaction_output_covariance(
# #                 final_models[seed1], 
# #                 final_models[seed2], 
# #                 layer_idx=0
# #             ).cpu().numpy()
# #             cross_covs[(seed1, seed2)] = cross_cov

# # #%%
# # # Visualize cross-model covariances
# # fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# # pairs = [(42, 123), (42, 456), (123, 456)]

# # # Row 1: With diagonal
# # vmax_with = max([np.abs(cross_covs[pair]).max() for pair in cross_covs.keys()])

# # for idx, (seed1, seed2) in enumerate(pairs):
# #     im = axes[0, idx].imshow(cross_covs[(seed1, seed2)], cmap='RdBu_r', 
# #                              vmin=-vmax_with, vmax=vmax_with)
# #     axes[0, idx].set_title(f'Cross-Covariance\nSeed {seed1} vs Seed {seed2}\nWith Diagonal')
# #     axes[0, idx].set_xlabel(f'Seed {seed2} Output Class')
# #     axes[0, idx].set_ylabel(f'Seed {seed1} Output Class')
# #     plt.colorbar(im, ax=axes[0, idx])

# # # Row 2: Without diagonal (for off-diagonal patterns)
# # cross_covs_no_diag = {}
# # for pair in cross_covs.keys():
# #     cov_viz = cross_covs[pair].copy()
# #     np.fill_diagonal(cov_viz, np.nan)
# #     cross_covs_no_diag[pair] = cov_viz

# # vmax_without = max([np.nanmax(np.abs(cross_covs_no_diag[pair])) for pair in cross_covs_no_diag.keys()])

# # for idx, (seed1, seed2) in enumerate(pairs):
# #     im = axes[1, idx].imshow(cross_covs_no_diag[(seed1, seed2)], cmap='RdBu_r', 
# #                              vmin=-vmax_without, vmax=vmax_without)
# #     axes[1, idx].set_title(f'Cross-Covariance\nSeed {seed1} vs Seed {seed2}\nWithout Diagonal')
# #     axes[1, idx].set_xlabel(f'Seed {seed2} Output Class')
# #     axes[1, idx].set_ylabel(f'Seed {seed1} Output Class')
# #     plt.colorbar(im, ax=axes[1, idx])

# # plt.tight_layout()
# # plt.savefig(figures_dir / "cross_model_covariances.png", dpi=300, bbox_inches='tight')
# # plt.show()

# # #%%
# # # Analyze diagonal vs off-diagonal
# # print("\n=== Cross-Covariance Analysis ===")

# # for seed1, seed2 in pairs:
# #     cross_cov = cross_covs[(seed1, seed2)]
    
# #     # Extract diagonal (same digit between models)
# #     diagonal = np.diag(cross_cov)
    
# #     # Extract off-diagonal
# #     off_diag_mask = ~np.eye(10, dtype=bool)
# #     off_diagonal = cross_cov[off_diag_mask]
    
# #     print(f"\nSeed {seed1} vs Seed {seed2}:")
# #     print(f"  Diagonal mean:     {diagonal.mean():>8.4f} (same digit alignment)")
# #     print(f"  Diagonal std:      {diagonal.std():>8.4f}")
# #     print(f"  Off-diagonal mean: {off_diagonal.mean():>8.4f} (cross-digit)")
# #     print(f"  Off-diagonal std:  {off_diagonal.std():>8.4f}")
# #     print(f"  Diagonal/Off-diag ratio: {abs(diagonal.mean() / off_diagonal.mean()):>8.2f}x")

# # #%%
# # # Compare self-covariance vs cross-covariance
# # print("\n=== Self vs Cross Covariance Comparison ===")

# # # For each seed, compare its self-covariance diagonal to cross-covariance diagonals
# # for seed in seeds:
# #     self_diag = np.diag(covs[seed])
    
# #     print(f"\nSeed {seed} self-covariance diagonal: {self_diag}")
    
# #     for other_seed in seeds:
# #         if other_seed != seed:
# #             cross_diag = np.diag(cross_covs[(seed, other_seed)])
# #             correlation = np.corrcoef(self_diag, cross_diag)[0, 1]
# #             print(f"  vs Seed {other_seed}: correlation = {correlation:.4f}")
# #             print(f"    Cross diagonal: {cross_diag}")
# # # %%




