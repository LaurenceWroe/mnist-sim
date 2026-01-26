#%%
"""
Compare three training setups:
1. Base: Train on all 10 digits from start
2. Partial: Train on 0-8, then add 9

Track checkpoints densely at start, compute cross-model similarities.
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
# Training function
def train_model(train_data, test_data, config, name, checkpoint_schedule):
    """
    Train a model and save checkpoints according to schedule.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        config: Model configuration
        name: Name for this training run
        checkpoint_schedule: Dict with 'first_n_batches', 'first_interval', 'later_interval'
    
    Returns:
        checkpoints: List of saved checkpoints
        history: Training history
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
    
    total_batch_count = 0
    first_n = checkpoint_schedule['first_n_batches']
    first_interval = checkpoint_schedule['first_interval']
    later_interval = checkpoint_schedule['later_interval']
    
    print(f"\n=== Training {name} ===")
    pbar = tqdm(range(config.epochs), desc=name)
    
    for epoch in pbar:
        for batch_idx, (x, y) in enumerate(loader):
            loss, acc = model.train().step(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_batch_count += 1
            
            # Checkpoint logic
            save_checkpoint = False
            if total_batch_count <= first_n:
                if total_batch_count % first_interval == 0:
                    save_checkpoint = True
            else:
                if (total_batch_count - first_n) % later_interval == 0:
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
    history['batch_steps'].append(total_batch_count)
    history['val_loss'].append(val_loss.item())
    history['val_acc'].append(val_acc.item())
    history['epochs'].append(config.epochs)
    
    checkpoints.append({
        'batch_step': total_batch_count,
        'epoch': config.epochs,
        'state_dict': deepcopy(model.state_dict()),
    })
    
    torch.set_grad_enabled(False)
    return checkpoints, history, model

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

checkpoint_schedule_base = {
    'first_n_batches': 500,
    'first_interval': 100,
    'later_interval': 250,
}

checkpoints_base, history_base, model_base = train_model(
    train_data_full,
    test_data_full,
    config_base,
    "Base (all digits)",
    checkpoint_schedule_base
)

#%%
# Setup 2: Partial model (train on 0-8, then add 9)
# Phase 1: Train on 0-8
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

checkpoint_schedule_partial_p1 = {
    'first_n_batches': 500,
    'first_interval': 100,
    'later_interval': 250,
}

checkpoints_partial_p1, history_partial_p1, model_partial = train_model(
    train_data_partial,
    test_data_partial,
    config_partial_p1,
    "Partial Phase 1 (digits 0-8)",
    checkpoint_schedule_partial_p1
)

phase1_final_batch = checkpoints_partial_p1[-1]['batch_step']

#%%
# Phase 2: Add digit 9
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

checkpoint_schedule_partial_p2 = {
    'first_n_batches': 500,
    'first_interval': 100,
    'later_interval': 250,
}

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

print("\n=== Partial Phase 2: Adding digit 9 ===")
pbar = tqdm(range(config_partial_p2.epochs), desc="Partial Phase 2")

total_batch_count_p2 = phase1_final_batch
phase2_batch_start = 0

for epoch in pbar:
    for batch_idx, (x, y) in enumerate(loader_p2):
        loss, acc = model_partial.train().step(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_batch_count_p2 += 1
        phase2_batch_start += 1
        
        # Checkpoint logic
        save_checkpoint = False
        if phase2_batch_start <= 500:
            if phase2_batch_start % 100 == 0:
                save_checkpoint = True
        else:
            if (phase2_batch_start - 500) % 250 == 0:
                save_checkpoint = True
        
        if save_checkpoint:
            val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)
            
            history_partial_p1['batch_steps'].append(total_batch_count_p2)
            history_partial_p1['val_loss'].append(val_loss.item())
            history_partial_p1['val_acc'].append(val_acc.item())
            history_partial_p1['epochs'].append(config_partial_p1.epochs + epoch + batch_idx / len(loader_p2))
            
            checkpoint = {
                'batch_step': total_batch_count_p2,
                'epoch': config_partial_p1.epochs + epoch + batch_idx / len(loader_p2),
                'state_dict': deepcopy(model_partial.state_dict()),
                'phase': 2,
            }
            checkpoints_partial_p1.append(checkpoint)
    
    scheduler.step()
    
    if epoch % 10 == 0:
        val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)
        pbar.set_postfix({'val_acc': f'{val_acc.item():.3f}'})

# Final checkpoint
val_loss, val_acc = model_partial.eval().step(test_data_full.x, test_data_full.y)
history_partial_p1['batch_steps'].append(total_batch_count_p2)
history_partial_p1['val_loss'].append(val_loss.item())
history_partial_p1['val_acc'].append(val_acc.item())
history_partial_p1['epochs'].append(config_partial_p1.epochs + config_partial_p2.epochs)

checkpoints_partial_p1.append({
    'batch_step': total_batch_count_p2,
    'epoch': config_partial_p1.epochs + config_partial_p2.epochs,
    'state_dict': deepcopy(model_partial.state_dict()),
    'phase': 2,
})

torch.set_grad_enabled(False)

# Rename for clarity
checkpoints_partial = checkpoints_partial_p1
history_partial = history_partial_p1

#%%
# Compute cross-model similarities
print("\nComputing cross-model similarities...")

# Get final models
final_base = Model(config_base)
final_base.load_state_dict(checkpoints_base[-1]['state_dict'])

final_partial = Model(config_partial_p2)
final_partial.load_state_dict(checkpoints_partial[-1]['state_dict'])

# Compute similarities
sims = {
    'base_to_base_final': [],
    'base_to_partial_final': [],
    'partial_to_partial_final': [],
    'partial_to_base_final': [],
}

print("Base checkpoints...")
for cp in tqdm(checkpoints_base):
    model_temp = Model(config_base)
    model_temp.load_state_dict(cp['state_dict'])
    
    sims['base_to_base_final'].append(
        tensor_similarity_single_layer(model_temp, final_base, layer_idx=0)
    )
    sims['base_to_partial_final'].append(
        tensor_similarity_single_layer(model_temp, final_partial, layer_idx=0)
    )

print("Partial checkpoints...")
for cp in tqdm(checkpoints_partial):
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
base_batch_steps = [cp['batch_step'] for cp in checkpoints_base]
axes[0, 0].plot(base_batch_steps, sims['base_to_base_final'], 
               label='Base → Base final', linewidth=2, marker='o', markersize=3)
axes[0, 0].plot(base_batch_steps, sims['base_to_partial_final'], 
               label='Base → Partial final', linewidth=2, marker='s', markersize=3)
axes[0, 0].set_xlabel('Batch Steps')
axes[0, 0].set_ylabel('Tensor Similarity')
axes[0, 0].set_title('Base Model Training')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Partial model similarities
partial_batch_steps = [cp['batch_step'] for cp in checkpoints_partial]
axes[0, 1].plot(partial_batch_steps, sims['partial_to_partial_final'], 
               label='Partial → Partial final', linewidth=2, marker='o', markersize=3)
axes[0, 1].plot(partial_batch_steps, sims['partial_to_base_final'], 
               label='Partial → Base final', linewidth=2, marker='s', markersize=3)
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
print(f"Partial final → Base final: {tensor_similarity_single_layer(final_partial, final_base):.4f}")