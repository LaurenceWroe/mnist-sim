"""
Utility functions for training with similarity tracking.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from .model import Model, Config, _collator
from .tn_sim import tensor_similarity_single_layer


def train_with_similarity_tracking(
    model,
    train_dataset,
    test_dataset,
    save_dir="checkpoints",
    checkpoint_every=5,
    transform=None,
):
    """
    Train model while tracking tensor similarity at checkpoints.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        save_dir: Directory to save checkpoints
        checkpoint_every: Save checkpoint every N epochs
        transform: Optional transform for data
    
    Returns:
        history: Dict with training metrics and similarities
        checkpoints: List of saved model state dicts
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    config = model.config
    torch.manual_seed(config.seed)
    torch.set_grad_enabled(True)
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=_collator(transform)
    )
    test_x, test_y = test_dataset.x, test_dataset.y
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'sim_to_final': [],
        'epochs': [],
    }
    
    checkpoints = []
    
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
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.3f}',
            'val_loss': f'{val_loss.item():.3f}',
            'val_acc': f'{val_acc.item():.3f}',
        })
    
    torch.set_grad_enabled(False)
    
    # Now compute similarity to final model for all checkpoints
    print("\nComputing similarity to final model...")
    final_state = checkpoints[-1]['state_dict']
    final_model = Model(config)
    final_model.load_state_dict(final_state)
    
    for cp in tqdm(checkpoints, desc="Computing similarities"):
        checkpoint_model = Model(config)
        checkpoint_model.load_state_dict(cp['state_dict'])
        sim = tensor_similarity_single_layer(checkpoint_model, final_model, layer_idx=0)
        history['sim_to_final'].append(sim)
    
    return history, checkpoints

def compute_checkpoint_similarity_matrix(checkpoints, config):
    """
    Compute pairwise similarity matrix between all checkpoints.
    
    Args:
        checkpoints: List of checkpoint dicts with 'epoch' and 'state_dict'
        config: Model config
    
    Returns:
        sim_matrix: (n_checkpoints, n_checkpoints) similarity matrix
        epochs: List of epoch numbers
    """
    n = len(checkpoints)
    sim_matrix = np.zeros((n, n))
    epochs = [cp['epoch'] for cp in checkpoints]
    
    # Load all checkpoint models
    models = []
    for cp in tqdm(checkpoints, desc="Loading checkpoints"):
        model = Model(config)
        model.load_state_dict(cp['state_dict'])
        models.append(model)
    
    # Compute pairwise similarities
    for i in tqdm(range(n), desc="Computing similarity matrix"):
        for j in range(n):
            if j < i:
                # Matrix is symmetric
                sim_matrix[i, j] = sim_matrix[j, i]
            else:
                sim = tensor_similarity_single_layer(models[i], models[j], layer_idx=0)
                sim_matrix[i, j] = sim
    
    return sim_matrix, epochs


def plot_training_metrics(history, save_path=None):
    """Plot training curves with similarity to final model."""
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
    
    # Similarity to final - only plot for checkpoint epochs
    checkpoint_epochs = [cp['epoch'] for cp in range(len(history['sim_to_final']))]
    axes[2].plot(history['sim_to_final'], linewidth=2, color='purple')
    axes[2].set_xlabel('Checkpoint Index')
    axes[2].set_ylabel('Tensor Similarity')
    axes[2].set_title('Similarity to Final Model')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def plot_checkpoint_similarity_heatmap(sim_matrix, epochs, save_path=None):
    """Plot heatmap of checkpoint similarities."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        sim_matrix,
        xticklabels=epochs,
        yticklabels=epochs,
        cmap='viridis',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Tensor Similarity'},
        ax=ax,
        square=True,
    )
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Epoch')
    ax.set_title('Pairwise Tensor Similarity Between Checkpoints')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
