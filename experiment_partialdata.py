#%%
"""
Compare different training curricula:
- Base: All 10 digits from start
- Partial (drop 8): Train on 0-7,9 then add 8
- Partial (drop 0): Train on 1-9 then add 0  
- Partial (drop 0,8): Train on 1-7,9 then add 0,8

Track covariance similarity to understand curriculum effects.
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

# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

#%%
# Load data for different digit subsets
print("Loading MNIST data...")
train_data_full = MNIST(train=True, download=True, device=device)
test_data_full = MNIST(train=False, download=True, device=device)

# Subsets
train_data_drop8 = MNIST(train=True, download=True, device=device, 
                         digits=[0,1,2,3,4,5,6,7,9])
test_data_drop8 = MNIST(train=False, download=True, device=device, 
                        digits=[0,1,2,3,4,5,6,7,9])

train_data_drop0 = MNIST(train=True, download=True, device=device, 
                         digits=[1,2,3,4,5,6,7,8,9])
test_data_drop0 = MNIST(train=False, download=True, device=device, 
                        digits=[1,2,3,4,5,6,7,8,9])

train_data_drop0and8 = MNIST(train=True, download=True, device=device, 
                             digits=[1,2,3,4,5,6,7,9])
test_data_drop0and8 = MNIST(train=False, download=True, device=device, 
                            digits=[1,2,3,4,5,6,7,9])

#%%
# Training function with initialization checkpoint
def train_model_curriculum(train_data, test_data, config, name):
    """
    Train with sparse checkpoints: 0, 10, 20, 50, 100, 200, ..., 500, then every 250
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
    
    checkpoints = []
    history = {
        'batch_steps': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': [],
    }
    
    # Save initialization
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = model.step(test_data.x, test_data.y)
    
    init_checkpoint = {
        'batch_step': 0,
        'epoch': 0,
        'state_dict': deepcopy(model.state_dict()),
    }
    checkpoints.append(init_checkpoint)
    
    history['batch_steps'].append(0)
    history['val_loss'].append(val_loss.item())
    history['val_acc'].append(val_acc.item())
    history['epochs'].append(0)
    
    total_batch_count = 0
    
    print(f"\n=== Training {name} ===")
    pbar = tqdm(range(config.epochs), desc=name)
    
    for epoch in pbar:
        for batch_idx, (x, y) in enumerate(loader):
            loss, acc = model.train().step(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_batch_count += 1
            
            # Sparse checkpoints
            save_checkpoint = False
            if total_batch_count in [10, 20, 50]:
                save_checkpoint = True
            elif total_batch_count <= 500 and total_batch_count % 100 == 0:
                save_checkpoint = True
            elif total_batch_count > 500 and (total_batch_count - 500) % 250 == 0:
                save_checkpoint = True
            
            if save_checkpoint:
                val_loss, val_acc = model.eval().step(test_data.x, test_data.y)
                
                history['batch_steps'].append(total_batch_count)
                history['val_loss'].append(val_loss.item())
                history['val_acc'].append(val_acc.item())
                history['epochs'].append(epoch + batch_idx / len(loader))
                
                checkpoint = {
                    'batch_step': total_batch_count,
                    'epoch': epoch + batch_idx / len(loader),
                    'state_dict': deepcopy(model.state_dict()),
                }
                checkpoints.append(checkpoint)
        
        scheduler.step()
        
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
    checkpoints.append(final_checkpoint)
    
    history['batch_steps'].append(total_batch_count)
    history['val_loss'].append(val_loss.item())
    history['val_acc'].append(val_acc.item())
    history['epochs'].append(config.epochs)
    
    torch.set_grad_enabled(False)
    return checkpoints, history, model

#%%
# Train base model (all 10 digits)
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

checkpoints_base, history_base, model_base = train_model_curriculum(
    train_data_full,
    test_data_full,
    config_base,
    "Base (all 10 digits)"
)

#%%
# Setup for partial models - Phase 1
curriculum_setups = {
    'drop8': {
        'train_data': train_data_drop8,
        'test_data': test_data_drop8,
        'name': 'Partial (drop 8)',
        'dropped_digits': [8],
    },
    'drop0': {
        'train_data': train_data_drop0,
        'test_data': test_data_drop0,
        'name': 'Partial (drop 0)',
        'dropped_digits': [0],
    },
    'drop0and8': {
        'train_data': train_data_drop0and8,
        'test_data': test_data_drop0and8,
        'name': 'Partial (drop 0,8)',
        'dropped_digits': [0, 8],
    },
}

# Train phase 1 for all partial models
partial_models = {}

for key, setup in curriculum_setups.items():
    config_p1 = Config(
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
    
    checkpoints_p1, history_p1, model_p1 = train_model_curriculum(
        setup['train_data'],
        setup['test_data'],
        config_p1,
        f"{setup['name']} - Phase 1"
    )
    
    partial_models[key] = {
        'checkpoints_p1': checkpoints_p1,
        'history_p1': history_p1,
        'model_p1': model_p1,
        'phase1_final_batch': checkpoints_p1[-1]['batch_step'],
        'config_p1': config_p1,
    }

#%%
# Phase 2: Add back the dropped digits
for key, setup in curriculum_setups.items():
    print(f"\n=== Phase 2 for {setup['name']}: Adding back digits {setup['dropped_digits']} ===")
    
    config_p2 = Config(
        lr=1e-4,  # Lower LR for fine-tuning
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
    
    # Continue from phase 1 model
    model_p2 = partial_models[key]['model_p1']
    
    torch.manual_seed(config_p2.seed)
    torch.set_grad_enabled(True)
    
    optimizer = AdamW(model_p2.parameters(), lr=config_p2.lr, weight_decay=config_p2.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=config_p2.epochs)
    
    loader_p2 = DataLoader(
        train_data_full,
        batch_size=config_p2.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collator(None)
    )
    
    checkpoints_p2 = []
    history_p2 = {
        'batch_steps': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': [],
    }
    
    phase1_final_batch = partial_models[key]['phase1_final_batch']
    total_batch_count = phase1_final_batch
    phase2_batch_count = 0
    
    pbar = tqdm(range(config_p2.epochs), desc=f"{setup['name']} - Phase 2")
    
    for epoch in pbar:
        for batch_idx, (x, y) in enumerate(loader_p2):
            loss, acc = model_p2.train().step(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_batch_count += 1
            phase2_batch_count += 1
            
            # Sparse checkpoints
            save_checkpoint = False
            if phase2_batch_count in [10, 20, 50]:
                save_checkpoint = True
            elif phase2_batch_count <= 500 and phase2_batch_count % 100 == 0:
                save_checkpoint = True
            elif phase2_batch_count > 500 and (phase2_batch_count - 500) % 250 == 0:
                save_checkpoint = True
            
            if save_checkpoint:
                val_loss, val_acc = model_p2.eval().step(test_data_full.x, test_data_full.y)
                
                history_p2['batch_steps'].append(total_batch_count)
                history_p2['val_loss'].append(val_loss.item())
                history_p2['val_acc'].append(val_acc.item())
                history_p2['epochs'].append(config_p1.epochs + epoch + batch_idx / len(loader_p2))
                
                checkpoint = {
                    'batch_step': total_batch_count,
                    'epoch': config_p1.epochs + epoch + batch_idx / len(loader_p2),
                    'state_dict': deepcopy(model_p2.state_dict()),
                    'phase': 2,
                }
                checkpoints_p2.append(checkpoint)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            val_loss, val_acc = model_p2.eval().step(test_data_full.x, test_data_full.y)
            pbar.set_postfix({'val_acc': f'{val_acc.item():.3f}'})
    
    # Save final
    val_loss, val_acc = model_p2.eval().step(test_data_full.x, test_data_full.y)
    
    final_checkpoint = {
        'batch_step': total_batch_count,
        'epoch': config_p1.epochs + config_p2.epochs,
        'state_dict': deepcopy(model_p2.state_dict()),
        'phase': 2,
    }
    checkpoints_p2.append(final_checkpoint)
    
    history_p2['batch_steps'].append(total_batch_count)
    history_p2['val_loss'].append(val_loss.item())
    history_p2['val_acc'].append(val_acc.item())
    history_p2['epochs'].append(config_p1.epochs + config_p2.epochs)
    
    torch.set_grad_enabled(False)
    
    # Store phase 2 results
    partial_models[key]['checkpoints_p2'] = checkpoints_p2
    partial_models[key]['history_p2'] = history_p2
    partial_models[key]['model_p2'] = model_p2
    partial_models[key]['config_p2'] = config_p2

#%%
# Extract key models for covariance analysis
print("\n=== Extracting models for covariance analysis ===")

# Base final
model_base_final = Model(config_base)
model_base_final.load_state_dict(checkpoints_base[-1]['state_dict'])

# All partial models - phase 1 and phase 2 finals
partial_final_models = {}

for key in partial_models.keys():
    # Phase 1 final
    model_p1_final = Model(partial_models[key]['config_p1'])
    model_p1_final.load_state_dict(partial_models[key]['checkpoints_p1'][-1]['state_dict'])
    
    # Phase 2 final
    model_p2_final = Model(partial_models[key]['config_p2'])
    model_p2_final.load_state_dict(partial_models[key]['checkpoints_p2'][-1]['state_dict'])
    
    partial_final_models[key] = {
        'phase1': model_p1_final,
        'phase2': model_p2_final,
    }

print("Models extracted successfully!")
print(f"  Base final")
for key in partial_models.keys():
    print(f"  {curriculum_setups[key]['name']}: Phase 1 final, Phase 2 final")

#%%
# Compute all covariance matrices
print("\n=== Computing covariance matrices ===")

# Base
cov_base = interaction_output_covariance(model_base_final, model_base_final, layer_idx=0).cpu().numpy()

# Partial models
covs_partial = {}
for key in partial_models.keys():
    covs_partial[f"{key}_p1"] = interaction_output_covariance(
        partial_final_models[key]['phase1'], 
        partial_final_models[key]['phase1'], 
        layer_idx=0
    ).cpu().numpy()
    
    covs_partial[f"{key}_p2"] = interaction_output_covariance(
        partial_final_models[key]['phase2'], 
        partial_final_models[key]['phase2'], 
        layer_idx=0
    ).cpu().numpy()

print("Covariance matrices computed!")
print(f"  Base: {cov_base.shape}")
for key in covs_partial.keys():
    print(f"  {key}: {covs_partial[key].shape}")





#%%
# Visualize all covariance matrices
print("\n=== Visualizing covariance matrices ===")

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Prepare all matrices for visualization
all_covs = {
    'Base': cov_base,
    'Drop 8 - P1': covs_partial['drop8_p1'],
    'Drop 8 - P2': covs_partial['drop8_p2'],
    'Drop 0 - P1': covs_partial['drop0_p1'],
    'Drop 0 - P2': covs_partial['drop0_p2'],
    'Drop 0,8 - P1': covs_partial['drop0and8_p1'],
    'Drop 0,8 - P2': covs_partial['drop0and8_p2'],
}

# Row 1: With diagonal
vmax_with = max([np.abs(cov).max() for cov in all_covs.values()])

titles = list(all_covs.keys())
for idx, (name, cov) in enumerate(all_covs.items()):
    row = idx // 4
    col = idx % 4
    
    if idx < 7:  # We have 7 plots (base + 3 curricula × 2 phases)
        im = axes[row, col].imshow(cov, cmap='RdBu_r', vmin=-vmax_with, vmax=vmax_with)
        axes[row, col].set_title(f'{name}\nWith Diagonal')
        axes[row, col].set_xlabel('Output Class')
        axes[row, col].set_ylabel('Output Class')
        plt.colorbar(im, ax=axes[row, col])
    else:
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(figures_dir / "curriculum_covariances_with_diagonal.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Without diagonal for better off-diagonal visibility
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

all_covs_no_diag = {}
for name, cov in all_covs.items():
    cov_viz = cov.copy()
    np.fill_diagonal(cov_viz, np.nan)
    all_covs_no_diag[name] = cov_viz

vmax_without = max([np.nanmax(np.abs(cov)) for cov in all_covs_no_diag.values()])

for idx, (name, cov) in enumerate(all_covs_no_diag.items()):
    row = idx // 4
    col = idx % 4
    
    if idx < 7:
        im = axes[row, col].imshow(cov, cmap='RdBu_r', vmin=-vmax_without, vmax=vmax_without)
        axes[row, col].set_title(f'{name}\nWithout Diagonal')
        axes[row, col].set_xlabel('Output Class')
        axes[row, col].set_ylabel('Output Class')
        plt.colorbar(im, ax=axes[row, col])
    else:
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(figures_dir / "curriculum_covariances_without_diagonal.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Compute covariance similarities between all models
print("\n=== Computing covariance similarities ===")

def covariance_similarity(cov1, cov2):
    """Compute similarity between two covariance matrices."""
    cov1_flat = cov1.flatten()
    cov2_flat = cov2.flatten()
    
    dot_product = np.dot(cov1_flat, cov2_flat)
    norm1 = np.linalg.norm(cov1_flat)
    norm2 = np.linalg.norm(cov2_flat)
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Compute similarity of all partial models to base
similarities_to_base = {}
for name, cov in all_covs.items():
    if name != 'Base':
        sim = covariance_similarity(cov, cov_base)
        similarities_to_base[name] = sim

print("\nCovariance similarity to Base:")
for name, sim in similarities_to_base.items():
    print(f"  {name:<20}: {sim:.4f}")

#%%
# Compare Phase 1 vs Phase 2 for each curriculum
print("\n=== Phase 1 vs Phase 2 Comparison ===")

for key in ['drop8', 'drop0', 'drop0and8']:
    p1_cov = covs_partial[f'{key}_p1']
    p2_cov = covs_partial[f'{key}_p2']
    
    sim = covariance_similarity(p1_cov, p2_cov)
    
    print(f"\n{curriculum_setups[key]['name']}:")
    print(f"  Phase 1 → Phase 2 similarity: {sim:.4f}")
    print(f"  Phase 1 → Base similarity: {similarities_to_base[f'{curriculum_setups[key][\"name\"].split(\"(\")[0].strip()} - P1']:.4f}")
    print(f"  Phase 2 → Base similarity: {similarities_to_base[f'{curriculum_setups[key][\"name\"].split(\"(\")[0].strip()} - P2']:.4f}")

#%%
# Visualize difference matrices (Partial - Base)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

comparisons = [
    ('Drop 8 - P1', 'Phase 1 (no digit 8)'),
    ('Drop 8 - P2', 'Phase 2 (added digit 8)'),
    ('Drop 0 - P1', 'Phase 1 (no digit 0)'),
    ('Drop 0 - P2', 'Phase 2 (added digit 0)'),
    ('Drop 0,8 - P1', 'Phase 1 (no digits 0,8)'),
    ('Drop 0,8 - P2', 'Phase 2 (added digits 0,8)'),
]

all_diffs = []
for name, _ in comparisons:
    diff = all_covs_no_diag[name] - all_covs_no_diag['Base']
    all_diffs.append(diff)

vmax_diff = max([np.nanmax(np.abs(diff)) for diff in all_diffs])

for idx, ((name, title), diff) in enumerate(zip(comparisons, all_diffs)):
    row = idx // 3
    col = idx % 3
    
    im = axes[row, col].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[row, col].set_title(f'{title}\nDifference from Base')
    axes[row, col].set_xlabel('Output Class')
    axes[row, col].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[row, col])

plt.tight_layout()
plt.savefig(figures_dir / "curriculum_differences_from_base.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Compare specific digit relationships
print("\n=== Digit Pair Covariances Across Curricula ===")

digit_pairs = [(0, 6), (0, 8), (1, 4), (3, 4), (6, 8), (6, 9), (8, 9)]

print(f"\n{'Pair':<10} {'Base':<10} {'Drop8-P1':<10} {'Drop8-P2':<10} {'Drop0-P1':<10} {'Drop0-P2':<10} {'Drop08-P1':<10} {'Drop08-P2':<10}")
print("-" * 90)

for d1, d2 in digit_pairs:
    values = [
        cov_base[d1, d2],
        covs_partial['drop8_p1'][d1, d2],
        covs_partial['drop8_p2'][d1, d2],
        covs_partial['drop0_p1'][d1, d2],
        covs_partial['drop0_p2'][d1, d2],
        covs_partial['drop0and8_p1'][d1, d2],
        covs_partial['drop0and8_p2'][d1, d2],
    ]
    
    print(f"{d1}-{d2:<8} {values[0]:>9.4f} {values[1]:>9.4f} {values[2]:>9.4f} {values[3]:>9.4f} {values[4]:>9.4f} {values[5]:>9.4f} {values[6]:>9.4f}")

#%%
# Focus on relationships with dropped digits
print("\n=== Relationships with Dropped Digits ===")

# Digit 8 relationships
print("\nDigit 8 relationships (when 8 is dropped):")
print(f"{'Other Digit':<15} {'Base':<10} {'Drop8-P1':<10} {'Drop8-P2':<10}")
print("-" * 45)
for d in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
    print(f"{d} vs 8: {cov_base[d, 8]:>9.4f} {covs_partial['drop8_p1'][d, 8]:>9.4f} {covs_partial['drop8_p2'][d, 8]:>9.4f}")

# Digit 0 relationships
print("\nDigit 0 relationships (when 0 is dropped):")
print(f"{'Other Digit':<15} {'Base':<10} {'Drop0-P1':<10} {'Drop0-P2':<10}")
print("-" * 45)
for d in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    print(f"{d} vs 0: {cov_base[d, 0]:>9.4f} {covs_partial['drop0_p1'][d, 0]:>9.4f} {covs_partial['drop0_p2'][d, 0]:>9.4f}")

#%%
# Diagonal values comparison
print("\n=== Diagonal Values (Self-representation) ===")

print(f"\n{'Digit':<10} {'Base':<10} {'Drop8-P1':<10} {'Drop8-P2':<10} {'Drop0-P1':<10} {'Drop0-P2':<10}")
print("-" * 70)

for d in range(10):
    values = [
        cov_base[d, d],
        covs_partial['drop8_p1'][d, d],
        covs_partial['drop8_p2'][d, d],
        covs_partial['drop0_p1'][d, d],
        covs_partial['drop0_p2'][d, d],
    ]
    
    print(f"{d:<10} {values[0]:>9.4f} {values[1]:>9.4f} {values[2]:>9.4f} {values[3]:>9.4f} {values[4]:>9.4f}")

#%%
# Track covariance similarity evolution
print("\n=== Computing covariance similarity evolution ===")

# For each partial model, compute similarity to base at each checkpoint
cov_sims_evolution = {}

for key in ['drop8', 'drop0', 'drop0and8']:
    print(f"Processing {curriculum_setups[key]['name']}...")
    
    # Phase 1
    sims_p1 = []
    for cp in tqdm(partial_models[key]['checkpoints_p1'], desc=f"{key} Phase 1"):
        model_temp = Model(partial_models[key]['config_p1'])
        model_temp.load_state_dict(cp['state_dict'])
        cov_temp = interaction_output_covariance(model_temp, model_temp, layer_idx=0).cpu().numpy()
        sim = covariance_similarity(cov_temp, cov_base)
        sims_p1.append(sim)
    
    # Phase 2
    sims_p2 = []
    for cp in tqdm(partial_models[key]['checkpoints_p2'], desc=f"{key} Phase 2"):
        model_temp = Model(partial_models[key]['config_p2'])
        model_temp.load_state_dict(cp['state_dict'])
        cov_temp = interaction_output_covariance(model_temp, model_temp, layer_idx=0).cpu().numpy()
        sim = covariance_similarity(cov_temp, cov_base)
        sims_p2.append(sim)
    
    cov_sims_evolution[key] = {
        'phase1': sims_p1,
        'phase2': sims_p2,
        'batch_steps_p1': [cp['batch_step'] for cp in partial_models[key]['checkpoints_p1']],
        'batch_steps_p2': [cp['batch_step'] for cp in partial_models[key]['checkpoints_p2']],
    }

#%%
# Plot covariance similarity evolution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, key in enumerate(['drop8', 'drop0', 'drop0and8']):
    data = cov_sims_evolution[key]
    
    # Phase 1
    axes[idx].plot(data['batch_steps_p1'], data['phase1'], 
                   label='Phase 1', linewidth=2, marker='o', markersize=4)
    
    # Phase 2
    axes[idx].plot(data['batch_steps_p2'], data['phase2'], 
                   label='Phase 2', linewidth=2, marker='s', markersize=4)
    
    # Mark phase transition
    phase1_final = partial_models[key]['phase1_final_batch']
    axes[idx].axvline(x=phase1_final, color='red', linestyle='--', 
                     label=f'Add {curriculum_setups[key]["dropped_digits"]}', linewidth=2)
    
    axes[idx].set_xlabel('Batch Steps')
    axes[idx].set_ylabel('Covariance Similarity to Base')
    axes[idx].set_title(f'{curriculum_setups[key]["name"]}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([0.85, 1.02])

plt.tight_layout()
plt.savefig(figures_dir / "curriculum_covariance_similarity_evolution.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Summary statistics
print("\n=== SUMMARY ===")
print("\nKey Findings:")

for key in ['drop8', 'drop0', 'drop0and8']:
    name = curriculum_setups[key]['name']
    dropped = curriculum_setups[key]['dropped_digits']
    
    sim_p1_init = cov_sims_evolution[key]['phase1'][0]
    sim_p1_final = cov_sims_evolution[key]['phase1'][-1]
    sim_p2_final = cov_sims_evolution[key]['phase2'][-1]
    
    print(f"\n{name} (dropped: {dropped}):")
    print(f"  Init → Base: {sim_p1_init:.4f}")
    print(f"  Phase 1 final → Base: {sim_p1_final:.4f}")
    print(f"  Phase 2 final → Base: {sim_p2_final:.4f}")
    print(f"  Recovery: {sim_p2_final - sim_p1_final:+.4f}")
    
    if sim_p2_final > 0.99:
        print(f"  ✓ Full recovery to base structure")
    elif sim_p2_final > sim_p1_final:
        print(f"  ⚠ Partial recovery ({(sim_p2_final - sim_p1_final) / (1 - sim_p1_final) * 100:.1f}% of gap closed)")
    else:
        print(f"  ✗ No recovery")

print("\nAnalysis complete!")
# %%
