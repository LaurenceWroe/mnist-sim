#%%
"""
Train single-layer bilinear model on MNIST with tensor similarity tracking.
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
import seaborn as sns

from functions.model import Model, Config, _collator
from functions.datasets import MNIST
from functions.tn_sim import tensor_similarity_single_layer, interaction_matrix_similarity

# import importlib
# from functions import tn_sim

# importlib.reload(tn_sim)
# from functions.tn_sim import interaction_matrix_similarity

device = "cpu"

#%%
# Load MNIST data
print("Loading MNIST data...")
train_data = MNIST(train=True, download=True, device=device)
test_data = MNIST(train=False, download=True, device=device)

#%%
# Create model
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
    seed=42,
)

model = Model(config)

#%%
# Training setup
save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True, parents=True)
checkpoint_every = 5

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
test_x, test_y = test_data.x, test_data.y

# History tracking
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'sim_to_final': [],
    'checkpoint_epochs': [],
    'epochs': [],
}

checkpoints = []

#%%
# Training loop
print("\nStarting training...")
pbar = tqdm(range(config.epochs))

for epoch in pbar:
    # Training
    epoch_metrics = []
    for x, y in loader:
        loss, acc = model.train().step(x, y)
        epoch_metrics.append((loss.item(), acc.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    val_loss, val_acc = model.eval().step(test_x, test_y)
    
    # Compute metrics
    train_loss = np.mean([m[0] for m in epoch_metrics])
    train_acc = np.mean([m[1] for m in epoch_metrics])
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss.item())
    history['val_acc'].append(val_acc.item())
    history['epochs'].append(epoch)
    
    # Save checkpoint
    if epoch % checkpoint_every == 0 or epoch == config.epochs - 1:
        checkpoint = deepcopy(model.state_dict())
        checkpoints.append({
            'epoch': epoch,
            'state_dict': checkpoint,
        })
        history['checkpoint_epochs'].append(epoch)
        torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Update progress bar
    pbar.set_postfix({
        'train_loss': f'{train_loss:.3f}',
        'val_loss': f'{val_loss.item():.3f}',
        'val_acc': f'{val_acc.item():.3f}',
    })

torch.set_grad_enabled(False)

#%%
# Compute similarity to final model for all checkpoints
print("\nComputing similarity to final model...")
final_state = checkpoints[-1]['state_dict']
final_model = Model(config)
final_model.load_state_dict(final_state)

for cp in tqdm(checkpoints, desc="Computing similarities"):
    checkpoint_model = Model(config)
    checkpoint_model.load_state_dict(cp['state_dict'])
    sim = tensor_similarity_single_layer(checkpoint_model, final_model, layer_idx=0)
    history['sim_to_final'].append(sim)

#%%
# Plot training metrics
print("\nPlotting training metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs = history['epochs']

# Loss curves
axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
axes[1].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Similarity to final
axes[2].plot(history['checkpoint_epochs'], history['sim_to_final'], 
             linewidth=2, color='purple', marker='o')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Tensor Similarity')
axes[2].set_title('Similarity to Final Model')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Compute checkpoint similarity matrix
print("\nComputing checkpoint similarity matrix...")
n = len(checkpoints)
sim_matrix = np.zeros((n, n))
checkpoint_epoch_list = [cp['epoch'] for cp in checkpoints]

# Load all checkpoint models
models = []
for cp in tqdm(checkpoints, desc="Loading checkpoints"):
    m = Model(config)
    m.load_state_dict(cp['state_dict'])
    models.append(m)

# Compute pairwise similarities
for i in tqdm(range(n), desc="Computing similarity matrix"):
    for j in range(n):
        if j < i:
            sim_matrix[i, j] = sim_matrix[j, i]
        else:
            sim = tensor_similarity_single_layer(models[i], models[j], layer_idx=0)
            sim_matrix[i, j] = sim

#%%
# Plot checkpoint similarity heatmap
print("\nPlotting checkpoint similarity heatmap...")
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(figsize=(10, 8))

# Create mask for upper triangle
mask = np.tril(np.ones_like(sim_matrix, dtype=bool), k=0)

sns.heatmap(
    sim_matrix,
    mask=mask,  # Only show lower triangle
    xticklabels=checkpoint_epoch_list,
    yticklabels=checkpoint_epoch_list,
    cmap='viridis',
    # norm=LogNorm(vmin=sim_matrix[~mask].min(), vmax=1),  # Log scale
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Tensor Similarity'},
    ax=ax,
    square=True,
)

ax.set_xlabel('Epoch')
ax.set_ylabel('Epoch')
ax.set_title('Pairwise Tensor Similarity Between Checkpoints')
ax.invert_yaxis() 

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("checkpoint_similarity_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nDone! Results saved.")

#%%

# Compute similarity to final model for all checkpoints (BOTH METHODS)
print("\nComputing similarity to final model...")


final_state = checkpoints[-1]['state_dict']
final_model = Model(config)
final_model.load_state_dict(final_state)

# Clear old similarities
history['sim_to_final'] = []
history['sim_to_final_interaction'] = []

history['sim_to_final_interaction'] = []  # New metric

for cp in tqdm(checkpoints, desc="Computing similarities"):
    checkpoint_model = Model(config)
    checkpoint_model.load_state_dict(cp['state_dict'])
    
    # Original method
    sim = tensor_similarity_single_layer(checkpoint_model, final_model, layer_idx=0)
    history['sim_to_final'].append(sim)
    
    # Interaction matrix method
    sim_int = interaction_matrix_similarity(checkpoint_model, final_model, layer_idx=0)
    history['sim_to_final_interaction'].append(sim_int)


# Plot training metrics (with both similarity methods)
print("\nPlotting training metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs = history['epochs']

# Loss curves
axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
axes[1].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Similarity to final - BOTH METHODS
axes[2].plot(history['checkpoint_epochs'], history['sim_to_final'], 
             linewidth=2, color='purple', marker='o', label='Original')
axes[2].plot(history['checkpoint_epochs'], history['sim_to_final_interaction'], 
             linewidth=2, color='orange', marker='s', label='Interaction Matrix')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Tensor Similarity')
axes[2].set_title('Similarity to Final Model')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
print("Comparison of methods:")
for i in range(min(5, len(history['sim_to_final']))):
    epoch = history['checkpoint_epochs'][i]
    orig = history['sim_to_final'][i]
    inter = history['sim_to_final_interaction'][i]
    diff = abs(orig - inter)
    print(f"Epoch {epoch}: Original={orig:.6f}, Interaction={inter:.6f}, Diff={diff:.8f}")




