#%%
"""
Compare training setups with dual checkpoint schedules:
- Dense checkpoints (every 5 batches, first 20 epochs) for similarity tracking
- Sparse checkpoints (100s then 250s) for covariance analysis

Setups:
1. Base: Train on all 10 digits from start
2. Partial: Train on 0-8, then add 9
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
from functions.tn_sim import tensor_similarity_single_layer
from functions.tn_covariance import interaction_output_covariance

device = "cpu"

#%%
# Load data
print("Loading MNIST data...")
train_data_partial = MNIST(train=True, download=True, device=device, digits=list(range(9)))
test_data_partial = MNIST(train=False, download=True, device=device, digits=list(range(9)))
train_data_full = MNIST(train=True, download=True, device=device)
test_data_full = MNIST(train=False, download=True, device=device)

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
    
    checkpoints_dense = []  # For similarity tracking
    checkpoints_sparse = []  # For covariance analysis
    
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
            
            # Dense checkpoints: every 5 batches for first 20 epochs
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
            
            # Sparse checkpoints: 0, 100, 200, ..., 500, then every 250
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
        
        # Continue tracking after epoch 20 (coarser)
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
    
    # Save final for both
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
# Setup 1: Base model (all digits)
config_base = Config(
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
    seed=42,
)

checkpoints_base_dense, checkpoints_base_sparse, history_base, model_base = train_model_dual_checkpoints(
    train_data_full,
    test_data_full,
    config_base,
    "Base (all digits)"
)

#%%
# Setup 2: Partial model - Phase 1 (train on 0-8)
config_partial_p1 = Config(
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
    seed=42,
)

checkpoints_partial_dense_p1, checkpoints_partial_sparse_p1, history_partial_p1, model_partial = train_model_dual_checkpoints(
    train_data_partial,
    test_data_partial,
    config_partial_p1,
    "Partial Phase 1 (digits 0-8)"
)

phase1_final_batch = checkpoints_partial_dense_p1[-1]['batch_step']

#%%
# Setup 2: Partial model - Phase 2 (add digit 9)
config_partial_p2 = Config(
    lr=1e-4,
    wd=0.5,
    epochs=50,
    batch_size=2048,
    d_hidden=256,
    n_layer=1,
    d_input=784,
    d_output=10,
    bias=False,
    residual=False,
    seed=42,
)

torch.manual_seed(config_partial_p2.seed)
torch.set_grad_enabled(True)

optimizer = AdamW(model_partial.parameters(), lr=config_partial_p2.lr, weight_decay=config_partial_p2.wd)
scheduler = CosineAnnealingLR(optimizer, T_max=config_partial_p2.epochs)

loader_p2 = DataLoader(
    train_data_full,
    batch_size=config_partial_p2.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=_collator(None)
)

batches_per_epoch = len(loader_p2)
dense_epoch_limit = 20

print("\n=== Partial Phase 2: Adding digit 9 ===")
pbar = tqdm(range(config_partial_p2.epochs), desc="Partial Phase 2")

total_batch_count_p2 = phase1_final_batch
phase2_batch_count = 0

for epoch in pbar:
    for batch_idx, (x, y) in enumerate(loader_p2):
        loss, acc = model_partial.train().step(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_batch_count_p2 += 1
        phase2_batch_count += 1
        
        # Dense checkpoints: every 5 batches for first 20 epochs
        if epoch < dense_epoch_limit and phase2_batch_count % 5 == 0:
            val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)
            
            history_partial_p1['batch_steps'].append(total_batch_count_p2)
            history_partial_p1['val_loss'].append(val_loss.item())
            history_partial_p1['val_acc'].append(val_acc.item())
            history_partial_p1['epochs'].append(config_partial_p1.epochs + epoch + batch_idx / batches_per_epoch)
            
            checkpoint = {
                'batch_step': total_batch_count_p2,
                'epoch': config_partial_p1.epochs + epoch + batch_idx / batches_per_epoch,
                'state_dict': deepcopy(model_partial.state_dict()),
                'phase': 2,
            }
            checkpoints_partial_dense_p1.append(checkpoint)
        
        # Sparse checkpoints
        save_sparse = False
        if phase2_batch_count <= 500 and phase2_batch_count % 100 == 0:
            save_sparse = True
        elif phase2_batch_count > 500 and (phase2_batch_count - 500) % 250 == 0:
            save_sparse = True
        
        if save_sparse:
            checkpoint_sparse = {
                'batch_step': total_batch_count_p2,
                'epoch': config_partial_p1.epochs + epoch + batch_idx / batches_per_epoch,
                'state_dict': deepcopy(model_partial.state_dict()),
                'phase': 2,
            }
            checkpoints_partial_sparse_p1.append(checkpoint_sparse)
    
    scheduler.step()
    
    # Continue tracking after epoch 20
    if epoch >= dense_epoch_limit and epoch % 5 == 0:
        val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)
        
        history_partial_p1['batch_steps'].append(total_batch_count_p2)
        history_partial_p1['val_loss'].append(val_loss.item())
        history_partial_p1['val_acc'].append(val_acc.item())
        history_partial_p1['epochs'].append(config_partial_p1.epochs + epoch)
        
        checkpoint = {
            'batch_step': total_batch_count_p2,
            'epoch': config_partial_p1.epochs + epoch,
            'state_dict': deepcopy(model_partial.state_dict()),
            'phase': 2,
        }
        checkpoints_partial_dense_p1.append(checkpoint)
    
    if epoch % 10 == 0:
        val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)
        pbar.set_postfix({'val_acc': f'{val_acc.item():.3f}'})

# Final checkpoint
val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)

final_checkpoint = {
    'batch_step': total_batch_count_p2,
    'epoch': config_partial_p1.epochs + config_partial_p2.epochs,
    'state_dict': deepcopy(model_partial.state_dict()),
    'phase': 2,
}

checkpoints_partial_dense_p1.append(final_checkpoint)
checkpoints_partial_sparse_p1.append(deepcopy(final_checkpoint))

history_partial_p1['batch_steps'].append(total_batch_count_p2)
history_partial_p1['val_loss'].append(val_loss.item())
history_partial_p1['val_acc'].append(val_acc.item())
history_partial_p1['epochs'].append(config_partial_p1.epochs + config_partial_p2.epochs)

torch.set_grad_enabled(False)

# Consolidate names
checkpoints_partial_dense = checkpoints_partial_dense_p1
checkpoints_partial_sparse = checkpoints_partial_sparse_p1
history_partial = history_partial_p1

#%%
# Compute cross-model similarities (using dense checkpoints)
print("\nComputing cross-model similarities...")

# Get final models
final_base = Model(config_base)
final_base.load_state_dict(checkpoints_base_dense[-1]['state_dict'])

final_partial = Model(config_partial_p2)
final_partial.load_state_dict(checkpoints_partial_dense[-1]['state_dict'])

# Compute similarities
sims = {
    'base_to_base_final': [],
    'base_to_partial_final': [],
    'partial_to_partial_final': [],
    'partial_to_base_final': [],
}

print("Base checkpoints...")
for cp in tqdm(checkpoints_base_dense):
    model_temp = Model(config_base)
    model_temp.load_state_dict(cp['state_dict'])
    
    sims['base_to_base_final'].append(
        tensor_similarity_single_layer(model_temp, final_base, layer_idx=0)
    )
    sims['base_to_partial_final'].append(
        tensor_similarity_single_layer(model_temp, final_partial, layer_idx=0)
    )

print("Partial checkpoints...")
for cp in tqdm(checkpoints_partial_dense):
    model_temp = Model(config_partial_p2)
    model_temp.load_state_dict(cp['state_dict'])
    
    sims['partial_to_partial_final'].append(
        tensor_similarity_single_layer(model_temp, final_partial, layer_idx=0)
    )
    sims['partial_to_base_final'].append(
        tensor_similarity_single_layer(model_temp, final_base, layer_idx=0)
    )

#%%
# Plot similarities
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Base model similarities
base_batch_steps = [cp['batch_step'] for cp in checkpoints_base_dense]
axes[0, 0].plot(base_batch_steps, sims['base_to_base_final'], 
               label='Base → Base final', linewidth=2, marker='o', markersize=2)
axes[0, 0].plot(base_batch_steps, sims['base_to_partial_final'], 
               label='Base → Partial final', linewidth=2, marker='s', markersize=2)
axes[0, 0].set_xlabel('Batch Steps')
axes[0, 0].set_ylabel('Tensor Similarity')
axes[0, 0].set_title('Base Model Training')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Partial model similarities
partial_batch_steps = [cp['batch_step'] for cp in checkpoints_partial_dense]
axes[0, 1].plot(partial_batch_steps, sims['partial_to_partial_final'], 
               label='Partial → Partial final', linewidth=2, marker='o', markersize=2)
axes[0, 1].plot(partial_batch_steps, sims['partial_to_base_final'], 
               label='Partial → Base final', linewidth=2, marker='s', markersize=2)
axes[0, 1].axvline(x=phase1_final_batch, color='red', linestyle='--', 
                   label='Add Digit 9', linewidth=2)
axes[0, 1].set_xlabel('Batch Steps')
axes[0, 1].set_ylabel('Tensor Similarity')
axes[0, 1].set_title('Partial Model Training')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Training curves
axes[1, 0].plot(history_base['batch_steps'], history_base['val_acc'], 
               label='Base', linewidth=2)
axes[1, 0].plot(history_partial['batch_steps'], history_partial['val_acc'], 
               label='Partial', linewidth=2)
axes[1, 0].axvline(x=phase1_final_batch, color='red', linestyle='--', 
                   label='Add Digit 9', linewidth=2)
axes[1, 0].set_xlabel('Batch Steps')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].set_title('Training Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# All similarities together
axes[1, 1].plot(base_batch_steps, sims['base_to_base_final'], 
               label='Base → Base final', linewidth=2, alpha=0.7)
axes[1, 1].plot(partial_batch_steps, sims['partial_to_partial_final'], 
               label='Partial → Partial final', linewidth=2, alpha=0.7)
axes[1, 1].axvline(x=phase1_final_batch, color='red', linestyle='--', 
                   label='Add Digit 9', linewidth=2)
axes[1, 1].set_xlabel('Batch Steps')
axes[1, 1].set_ylabel('Tensor Similarity')
axes[1, 1].set_title('Convergence Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cross_model_similarity_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal similarities:")
print(f"Base final → Partial final: {tensor_similarity_single_layer(final_base, final_partial):.4f}")
print(f"\nSparse checkpoints saved:")
print(f"  Base: {len(checkpoints_base_sparse)} checkpoints")
print(f"  Partial: {len(checkpoints_partial_sparse)} checkpoints")






#%%
# Compute covariance matrices for sparse checkpoints
print("\nComputing covariance matrices for sparse checkpoints...")

# Select key checkpoints to compare
# We'll look at: early, mid, late for both base and partial
key_batch_steps = [100, 500, 1000, 2000]  # Adjust based on what's available

# Get covariances for base model
base_covs = {}
for batch_step in key_batch_steps:
    # Find checkpoint closest to this batch step
    cp = min(checkpoints_base_sparse, key=lambda x: abs(x['batch_step'] - batch_step))
    if abs(cp['batch_step'] - batch_step) < 100:  # Only if within 100 batches
        model_temp = Model(config_base)
        model_temp.load_state_dict(cp['state_dict'])
        
        cov = interaction_output_covariance(model_temp, model_temp, layer_idx=0)
        base_covs[cp['batch_step']] = cov.cpu().numpy()

# Get covariances for partial model
partial_covs = {}
for batch_step in key_batch_steps + [phase1_final_batch, phase1_final_batch + 100, phase1_final_batch + 500]:
    cp = min(checkpoints_partial_sparse, key=lambda x: abs(x['batch_step'] - batch_step))
    if abs(cp['batch_step'] - batch_step) < 100:
        model_temp = Model(config_partial_p2)
        model_temp.load_state_dict(cp['state_dict'])
        
        cov = interaction_output_covariance(model_temp, model_temp, layer_idx=0)
        partial_covs[cp['batch_step']] = cov.cpu().numpy()

# Also get final covariances
base_covs['final'] = interaction_output_covariance(final_base, final_base, layer_idx=0).cpu().numpy()
partial_covs['final'] = interaction_output_covariance(final_partial, final_partial, layer_idx=0).cpu().numpy()

print(f"Base covariances computed for batch steps: {list(base_covs.keys())}")
print(f"Partial covariances computed for batch steps: {list(partial_covs.keys())}")

#%%
# Plot covariance evolution for base model
base_steps_sorted = sorted([k for k in base_covs.keys() if k != 'final'])
n_base = len(base_steps_sorted) + 1  # +1 for final

fig, axes = plt.subplots(1, n_base, figsize=(5*n_base, 5))
if n_base == 1:
    axes = [axes]

for idx, step in enumerate(base_steps_sorted + ['final']):
    cov = base_covs[step]
    
    # Mask diagonal for better visualization
    cov_no_diag = cov.copy()
    np.fill_diagonal(cov_no_diag, np.nan)
    vmax = np.nanmax(np.abs(cov_no_diag))
    
    im = axes[idx].imshow(cov_no_diag, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    if step == 'final':
        axes[idx].set_title(f'Base - Final')
    else:
        axes[idx].set_title(f'Base - Batch {step}')
    
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("covariance_evolution_base.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot covariance evolution for partial model
partial_steps_sorted = sorted([k for k in partial_covs.keys() if k != 'final'])
n_partial = len(partial_steps_sorted) + 1

fig, axes = plt.subplots(1, n_partial, figsize=(5*n_partial, 5))
if n_partial == 1:
    axes = [axes]

for idx, step in enumerate(partial_steps_sorted + ['final']):
    cov = partial_covs[step]
    
    cov_no_diag = cov.copy()
    np.fill_diagonal(cov_no_diag, np.nan)
    vmax = np.nanmax(np.abs(cov_no_diag))
    
    im = axes[idx].imshow(cov_no_diag, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    if step == 'final':
        title = f'Partial - Final'
    elif step > phase1_final_batch:
        title = f'Partial - Batch {step}\n(After adding 9)'
    else:
        title = f'Partial - Batch {step}\n(Before adding 9)'
    
    axes[idx].set_title(title)
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("covariance_evolution_partial.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Compare final covariance matrices: Base vs Partial
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Base final
cov_base_final = base_covs['final'].copy()
np.fill_diagonal(cov_base_final, np.nan)
vmax = max(np.nanmax(np.abs(cov_base_final)), np.nanmax(np.abs(partial_covs['final'])))

im0 = axes[0].imshow(cov_base_final, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0].set_title('Base Final (trained on all 10 digits)')
axes[0].set_xlabel('Output Class')
axes[0].set_ylabel('Output Class')
plt.colorbar(im0, ax=axes[0])

# Partial final
cov_partial_final = partial_covs['final'].copy()
np.fill_diagonal(cov_partial_final, np.nan)

im1 = axes[1].imshow(cov_partial_final, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_title('Partial Final (0-8 then added 9)')
axes[1].set_xlabel('Output Class')
axes[1].set_ylabel('Output Class')
plt.colorbar(im1, ax=axes[1])

# Difference
diff = cov_partial_final - cov_base_final
vmax_diff = np.nanmax(np.abs(diff))

im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[2].set_title('Difference (Partial - Base)')
axes[2].set_xlabel('Output Class')
axes[2].set_ylabel('Output Class')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("covariance_final_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Analyze specific digit relationships
print("\n=== Covariance Comparison: Base vs Partial ===")
print("\nFinal covariance values for key digit pairs:")

digit_pairs = [(0, 6), (1, 4), (3, 4), (0, 8), (3, 9), (0, 9)]

for d1, d2 in digit_pairs:
    base_val = base_covs['final'][d1, d2]
    partial_val = partial_covs['final'][d1, d2]
    diff_val = partial_val - base_val
    
    print(f"Digits {d1}-{d2}: Base={base_val:7.4f}, Partial={partial_val:7.4f}, Diff={diff_val:+7.4f}")

#%%
# Track how specific digit pair covariances evolve
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (d1, d2) in enumerate(digit_pairs):
    # Base evolution
    base_steps_sorted_numeric = sorted([k for k in base_covs.keys() if k != 'final'])
    base_values = [base_covs[step][d1, d2] for step in base_steps_sorted_numeric]
    base_values.append(base_covs['final'][d1, d2])
    
    axes[idx].plot(base_steps_sorted_numeric + [base_batch_steps[-1]], base_values, 
                   marker='o', label='Base', linewidth=2)
    
    # Partial evolution
    partial_steps_sorted_numeric = sorted([k for k in partial_covs.keys() if k != 'final'])
    partial_values = [partial_covs[step][d1, d2] for step in partial_steps_sorted_numeric]
    partial_values.append(partial_covs['final'][d1, d2])
    
    axes[idx].plot(partial_steps_sorted_numeric + [partial_batch_steps[-1]], partial_values, 
                   marker='s', label='Partial', linewidth=2)
    
    axes[idx].axvline(x=phase1_final_batch, color='red', linestyle='--', 
                     label='Add Digit 9', linewidth=1, alpha=0.5)
    
    axes[idx].set_xlabel('Batch Steps')
    axes[idx].set_ylabel(f'Covariance({d1}, {d2})')
    axes[idx].set_title(f'Digits {d1} vs {d2}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig("covariance_pair_evolution.png", dpi=300, bbox_inches='tight')
plt.show()
# %%





#%%
# Extract the three key models for comparison
print("\nExtracting models for comparison...")

# 1. Base final (trained on all 10 from start)
model_base_final = Model(config_base)
model_base_final.load_state_dict(checkpoints_base_sparse[-1]['state_dict'])

# 2. Partial phase 1 final (trained only on 0-8)
# Find the checkpoint right before phase 2 starts
phase1_final_cp = [cp for cp in checkpoints_partial_sparse if cp['batch_step'] == phase1_final_batch][0]
model_partial_p1_final = Model(config_partial_p1)
model_partial_p1_final.load_state_dict(phase1_final_cp['state_dict'])

# 3. Partial phase 2 final (after adding 9)
model_partial_p2_final = Model(config_partial_p2)
model_partial_p2_final.load_state_dict(checkpoints_partial_sparse[-1]['state_dict'])

#%%
# Compute covariance matrices for all three
print("Computing covariance matrices...")

cov_base = interaction_output_covariance(model_base_final, model_base_final, layer_idx=0).cpu().numpy()
cov_partial_p1 = interaction_output_covariance(model_partial_p1_final, model_partial_p1_final, layer_idx=0).cpu().numpy()
cov_partial_p2 = interaction_output_covariance(model_partial_p2_final, model_partial_p2_final, layer_idx=0).cpu().numpy()

#%%
# Visualize all three + differences
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: The three covariance matrices
vmax_global = max(np.abs(cov_base).max(), np.abs(cov_partial_p1).max(), np.abs(cov_partial_p2).max())

# Mask diagonals
cov_base_viz = cov_base.copy()
np.fill_diagonal(cov_base_viz, np.nan)

cov_p1_viz = cov_partial_p1.copy()
np.fill_diagonal(cov_p1_viz, np.nan)

cov_p2_viz = cov_partial_p2.copy()
np.fill_diagonal(cov_p2_viz, np.nan)

vmax = max(np.nanmax(np.abs(cov_base_viz)), np.nanmax(np.abs(cov_p1_viz)), np.nanmax(np.abs(cov_p2_viz)))

im0 = axes[0, 0].imshow(cov_base_viz, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0, 0].set_title('Base Final\n(All 10 digits from start)')
axes[0, 0].set_xlabel('Output Class')
axes[0, 0].set_ylabel('Output Class')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(cov_p1_viz, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0, 1].set_title('Partial Phase 1 Final\n(Only digits 0-8)')
axes[0, 1].set_xlabel('Output Class')
axes[0, 1].set_ylabel('Output Class')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(cov_p2_viz, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0, 2].set_title('Partial Phase 2 Final\n(After adding digit 9)')
axes[0, 2].set_xlabel('Output Class')
axes[0, 2].set_ylabel('Output Class')
plt.colorbar(im2, ax=axes[0, 2])

# Row 2: Differences
diff_p1_vs_base = cov_p1_viz - cov_base_viz
diff_p2_vs_base = cov_p2_viz - cov_base_viz
diff_p2_vs_p1 = cov_p2_viz - cov_p1_viz

vmax_diff = max(np.nanmax(np.abs(diff_p1_vs_base)), 
                np.nanmax(np.abs(diff_p2_vs_base)),
                np.nanmax(np.abs(diff_p2_vs_p1)))

im3 = axes[1, 0].imshow(diff_p1_vs_base, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[1, 0].set_title('Difference: Phase1 - Base\n(Effect of excluding digit 9)')
axes[1, 0].set_xlabel('Output Class')
axes[1, 0].set_ylabel('Output Class')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(diff_p2_vs_base, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[1, 1].set_title('Difference: Phase2 - Base\n(Final curriculum effect)')
axes[1, 1].set_xlabel('Output Class')
axes[1, 1].set_ylabel('Output Class')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(diff_p2_vs_p1, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[1, 2].set_title('Difference: Phase2 - Phase1\n(Effect of adding digit 9)')
axes[1, 2].set_xlabel('Output Class')
axes[1, 2].set_ylabel('Output Class')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("covariance_three_way_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Quantitative comparison
print("\n=== Quantitative Covariance Comparison ===")
print("\nDigit pair covariances:")
print(f"{'Pair':<10} {'Base':<10} {'Phase1':<10} {'Phase2':<10} {'P1-Base':<10} {'P2-Base':<10} {'P2-P1':<10}")
print("-" * 70)

digit_pairs = [(0, 6), (1, 4), (3, 4), (0, 8), (3, 9), (0, 9), (1, 9), (6, 9)]

for d1, d2 in digit_pairs:
    base_val = cov_base[d1, d2]
    p1_val = cov_partial_p1[d1, d2]
    p2_val = cov_partial_p2[d1, d2]
    
    diff_p1 = p1_val - base_val
    diff_p2 = p2_val - base_val
    diff_p2_p1 = p2_val - p1_val
    
    print(f"{d1}-{d2:<8} {base_val:>9.4f} {p1_val:>9.4f} {p2_val:>9.4f} {diff_p1:>+9.4f} {diff_p2:>+9.4f} {diff_p2_p1:>+9.4f}")

#%%
# Focus on digit 9 relationships
print("\n=== Digit 9 Relationships ===")
print(f"{'Other Digit':<15} {'Base':<10} {'Phase1':<10} {'Phase2':<10}")
print("-" * 45)

for d in range(9):
    print(f"{d} vs 9: {cov_base[d, 9]:>9.4f} {cov_partial_p1[d, 9]:>9.4f} {cov_partial_p2[d, 9]:>9.4f}")

#%%
# Visualize all three + differences (WITH and WITHOUT diagonal)
fig, axes = plt.subplots(3, 3, figsize=(18, 18))

# Row 1: With diagonal
vmax_with = max(np.abs(cov_base).max(), np.abs(cov_partial_p1).max(), np.abs(cov_partial_p2).max())

im0 = axes[0, 0].imshow(cov_base, cmap='RdBu_r', vmin=-vmax_with, vmax=vmax_with)
axes[0, 0].set_title('Base Final\n(All 10 digits from start)\nWith Diagonal')
axes[0, 0].set_xlabel('Output Class')
axes[0, 0].set_ylabel('Output Class')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(cov_partial_p1, cmap='RdBu_r', vmin=-vmax_with, vmax=vmax_with)
axes[0, 1].set_title('Partial Phase 1 Final\n(Only digits 0-8)\nWith Diagonal')
axes[0, 1].set_xlabel('Output Class')
axes[0, 1].set_ylabel('Output Class')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(cov_partial_p2, cmap='RdBu_r', vmin=-vmax_with, vmax=vmax_with)
axes[0, 2].set_title('Partial Phase 2 Final\n(After adding digit 9)\nWith Diagonal')
axes[0, 2].set_xlabel('Output Class')
axes[0, 2].set_ylabel('Output Class')
plt.colorbar(im2, ax=axes[0, 2])

# Row 2: Without diagonal (for better off-diagonal visibility)
cov_base_viz = cov_base.copy()
np.fill_diagonal(cov_base_viz, np.nan)

cov_p1_viz = cov_partial_p1.copy()
np.fill_diagonal(cov_p1_viz, np.nan)

cov_p2_viz = cov_partial_p2.copy()
np.fill_diagonal(cov_p2_viz, np.nan)

vmax_without = max(np.nanmax(np.abs(cov_base_viz)), np.nanmax(np.abs(cov_p1_viz)), np.nanmax(np.abs(cov_p2_viz)))

im3 = axes[1, 0].imshow(cov_base_viz, cmap='RdBu_r', vmin=-vmax_without, vmax=vmax_without)
axes[1, 0].set_title('Base Final\nWithout Diagonal')
axes[1, 0].set_xlabel('Output Class')
axes[1, 0].set_ylabel('Output Class')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(cov_p1_viz, cmap='RdBu_r', vmin=-vmax_without, vmax=vmax_without)
axes[1, 1].set_title('Partial Phase 1 Final\nWithout Diagonal')
axes[1, 1].set_xlabel('Output Class')
axes[1, 1].set_ylabel('Output Class')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(cov_p2_viz, cmap='RdBu_r', vmin=-vmax_without, vmax=vmax_without)
axes[1, 2].set_title('Partial Phase 2 Final\nWithout Diagonal')
axes[1, 2].set_xlabel('Output Class')
axes[1, 2].set_ylabel('Output Class')
plt.colorbar(im5, ax=axes[1, 2])

# Row 3: Differences (without diagonal)
diff_p1_vs_base = cov_p1_viz - cov_base_viz
diff_p2_vs_base = cov_p2_viz - cov_base_viz
diff_p2_vs_p1 = cov_p2_viz - cov_p1_viz

diff_p1_vs_base = cov_partial_p1 - cov_base
diff_p2_vs_base = cov_partial_p2 - cov_base
diff_p2_vs_p1 = cov_partial_p2 - cov_partial_p1

vmax_diff = max(np.nanmax(np.abs(diff_p1_vs_base)), 
                np.nanmax(np.abs(diff_p2_vs_base)),
                np.nanmax(np.abs(diff_p2_vs_p1)))

im6 = axes[2, 0].imshow(diff_p1_vs_base, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[2, 0].set_title('Difference: Phase1 - Base\n(Effect of excluding digit 9)')
axes[2, 0].set_xlabel('Output Class')
axes[2, 0].set_ylabel('Output Class')
plt.colorbar(im6, ax=axes[2, 0])

im7 = axes[2, 1].imshow(diff_p2_vs_base, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[2, 1].set_title('Difference: Phase2 - Base\n(Final curriculum effect)')
axes[2, 1].set_xlabel('Output Class')
axes[2, 1].set_ylabel('Output Class')
plt.colorbar(im7, ax=axes[2, 1])

im8 = axes[2, 2].imshow(diff_p2_vs_p1, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
axes[2, 2].set_title('Difference: Phase2 - Phase1\n(Effect of adding digit 9)')
axes[2, 2].set_xlabel('Output Class')
axes[2, 2].set_ylabel('Output Class')
plt.colorbar(im8, ax=axes[2, 2])

plt.tight_layout()
plt.savefig("covariance_three_way_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Also compare diagonal values specifically
print("\n=== Diagonal Values (Self-covariance) ===")
print(f"{'Digit':<10} {'Base':<12} {'Phase1':<12} {'Phase2':<12} {'P1-Base':<12} {'P2-Base':<12}")
print("-" * 70)

for d in range(10):
    base_diag = cov_base[d, d]
    p1_diag = cov_partial_p1[d, d]
    p2_diag = cov_partial_p2[d, d]
    
    diff_p1 = p1_diag - base_diag
    diff_p2 = p2_diag - base_diag
    
    print(f"{d:<10} {base_diag:>11.4f} {p1_diag:>11.4f} {p2_diag:>11.4f} {diff_p1:>+11.4f} {diff_p2:>+11.4f}")
# %%


#%%
# Extract embedding matrices from the three models
print("\nExtracting embedding matrices...")

W_embed_base = model_base_final.embed.weight.data.cpu().numpy()  # [256, 784]
W_embed_p1 = model_partial_p1_final.embed.weight.data.cpu().numpy()
W_embed_p2 = model_partial_p2_final.embed.weight.data.cpu().numpy()

print(f"Embedding shape: {W_embed_base.shape}")

#%%
# 1. Compare embedding similarity between models
from sklearn.decomposition import PCA

# Flatten for comparison or use cosine similarity per dimension
def embedding_similarity(W1, W2):
    """Compute average cosine similarity across hidden dimensions"""
    sims = []
    for i in range(W1.shape[0]):  # For each hidden dimension
        v1 = W1[i]
        v2 = W2[i]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        sims.append(sim)
    return np.array(sims)

sim_base_p1 = embedding_similarity(W_embed_base, W_embed_p1)
sim_base_p2 = embedding_similarity(W_embed_base, W_embed_p2)
sim_p1_p2 = embedding_similarity(W_embed_p1, W_embed_p2)

print(f"\nEmbedding similarity (per hidden dim, then averaged):")
print(f"Base vs Phase1: {sim_base_p1.mean():.4f} ± {sim_base_p1.std():.4f}")
print(f"Base vs Phase2: {sim_base_p2.mean():.4f} ± {sim_base_p2.std():.4f}")
print(f"Phase1 vs Phase2: {sim_p1_p2.mean():.4f} ± {sim_p1_p2.std():.4f}")

#%%
# 2. Visualize embedding space with PCA
# Project embedding vectors to 2D
pca = PCA(n_components=2)

# Transpose so each row is a hidden dimension's 784-d vector
W_all = np.vstack([W_embed_base, W_embed_p1, W_embed_p2])
W_2d = pca.fit_transform(W_all)

# Split back
W_base_2d = W_2d[:256]
W_p1_2d = W_2d[256:512]
W_p2_2d = W_2d[512:]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(W_base_2d[:, 0], W_base_2d[:, 1], alpha=0.5, s=20)
axes[0].set_title('Base Embedding (PCA)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(W_p1_2d[:, 0], W_p1_2d[:, 1], alpha=0.5, s=20, color='orange')
axes[1].set_title('Phase 1 Embedding (PCA)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(W_p2_2d[:, 0], W_p2_2d[:, 1], alpha=0.5, s=20, color='green')
axes[2].set_title('Phase 2 Embedding (PCA)')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("embedding_pca_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# 3. Analyze which hidden dimensions changed most
diff_p1_base = W_embed_p1 - W_embed_base
diff_p2_base = W_embed_p2 - W_embed_base
diff_p2_p1 = W_embed_p2 - W_embed_p1

# Frobenius norm per hidden dimension
norm_diff_p1 = np.linalg.norm(diff_p1_base, axis=1)  # [256]
norm_diff_p2 = np.linalg.norm(diff_p2_base, axis=1)
norm_diff_p2_p1 = np.linalg.norm(diff_p2_p1, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(range(256), norm_diff_p1)
axes[0].set_title('Embedding Change: Phase1 vs Base\n(per hidden dimension)')
axes[0].set_xlabel('Hidden Dimension')
axes[0].set_ylabel('L2 Norm of Difference')
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(256), norm_diff_p2)
axes[1].set_title('Embedding Change: Phase2 vs Base')
axes[1].set_xlabel('Hidden Dimension')
axes[1].set_ylabel('L2 Norm of Difference')
axes[1].grid(True, alpha=0.3)

axes[2].bar(range(256), norm_diff_p2_p1)
axes[2].set_title('Embedding Change: Phase2 vs Phase1\n(effect of adding digit 9)')
axes[2].set_xlabel('Hidden Dimension')
axes[2].set_ylabel('L2 Norm of Difference')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("embedding_dimension_changes.png", dpi=300, bbox_inches='tight')
plt.show()

# Find most changed dimensions
top_k = 10
top_changed_p2_p1 = np.argsort(norm_diff_p2_p1)[-top_k:][::-1]
print(f"\nTop {top_k} hidden dimensions that changed most when adding digit 9:")
print(top_changed_p2_p1)
print(f"Change magnitudes: {norm_diff_p2_p1[top_changed_p2_p1]}")

#%%
# 4. Visualize specific embedding vectors as 28x28 images
# Each hidden dimension has a 784-d "receptive field"
fig, axes = plt.subplots(3, top_k, figsize=(20, 6))

for idx, dim in enumerate(top_changed_p2_p1):
    # Base
    axes[0, idx].imshow(W_embed_base[dim].reshape(28, 28), cmap='RdBu_r', 
                        vmin=-W_embed_base[dim].max(), vmax=W_embed_base[dim].max())
    axes[0, idx].set_title(f'Dim {dim}')
    axes[0, idx].axis('off')
    
    # Phase 1
    axes[1, idx].imshow(W_embed_p1[dim].reshape(28, 28), cmap='RdBu_r',
                        vmin=-W_embed_p1[dim].max(), vmax=W_embed_p1[dim].max())
    axes[1, idx].axis('off')
    
    # Phase 2
    axes[2, idx].imshow(W_embed_p2[dim].reshape(28, 28), cmap='RdBu_r',
                        vmin=-W_embed_p2[dim].max(), vmax=W_embed_p2[dim].max())
    axes[2, idx].axis('off')

axes[0, 0].set_ylabel('Base', fontsize=12)
axes[1, 0].set_ylabel('Phase 1', fontsize=12)
axes[2, 0].set_ylabel('Phase 2', fontsize=12)

plt.suptitle('Top 10 Most Changed Embedding Dimensions (as 28x28 filters)', fontsize=14)
plt.tight_layout()
plt.savefig("embedding_receptive_fields.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# 5. Connect embedding changes to covariance changes
# For the dimensions that changed most, check if they correlate with 
# covariance changes for specific digits

print("\n=== Connecting Embeddings to Covariance ===")
print("Analyzing if embedding changes correlate with covariance changes...")

# This is speculative analysis - are the dimensions that changed most
# the ones involved in digit 9 relationships?