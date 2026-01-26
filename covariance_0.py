
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


# %%
#%%
# Compute covariance matrices for different epochs vs final
from functions.tn_covariance import interaction_output_covariance, interaction_latent_covariance

# Select epochs to compare with final
compare_epochs = [0, 20, 40, 60, 80,99]
final_epoch = 99

# Find the final checkpoint
final_checkpoint = [cp for cp in checkpoints if cp['epoch'] == final_epoch][0]
final_model = Model(config)
final_model.load_state_dict(final_checkpoint['state_dict'])

# Compute covariances
output_covs = {}
latent_covs = {}

for epoch in compare_epochs:
    # Find checkpoint for this epoch
    cp = [c for c in checkpoints if c['epoch'] == epoch][0]
    model_t = Model(config)
    model_t.load_state_dict(cp['state_dict'])
    
    # Compute covariances
    output_cov = interaction_output_covariance(model_t, final_model, layer_idx=0)
    latent_cov = interaction_latent_covariance(model_t, final_model, layer_idx=0)
    
    output_covs[epoch] = output_cov
    latent_covs[epoch] = latent_cov
    
    print(f"Epoch {epoch}: Output cov shape {output_cov.shape}, Latent cov shape {latent_cov.shape}")

#%%
# Visualize output covariances
fig, axes = plt.subplots(1, len(compare_epochs), figsize=(20, 4))

for idx, epoch in enumerate(compare_epochs):
    cov = output_covs[epoch].cpu().numpy()
    im = axes[idx].imshow(cov, cmap='RdBu_r', vmin=-cov.max(), vmax=cov.max())
    axes[idx].set_title(f'Epoch {epoch} vs {final_epoch}\nOutput Covariance')
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("output_covariances.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
#%%
# Visualize output covariances with log scale
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, len(compare_epochs), figsize=(20, 4))

for idx, epoch in enumerate(compare_epochs):
    cov = output_covs[epoch].cpu().numpy()
    # Get positive values only for log scale
    vmin = np.abs(cov[cov != 0]).min() if np.any(cov != 0) else 1e-10
    vmax = np.abs(cov).max()
    
    im = axes[idx].imshow(np.abs(cov), cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
    axes[idx].set_title(f'Epoch {epoch} vs {final_epoch}\nOutput Covariance (log scale)')
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("output_covariances_log.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Visualize output covariances - WITH diagonal
fig, axes = plt.subplots(1, len(compare_epochs), figsize=(20, 4))

for idx, epoch in enumerate(compare_epochs):
    cov = output_covs[epoch].cpu().numpy()
    vmax = np.abs(cov).max()
    
    im = axes[idx].imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[idx].set_title(f'Epoch {epoch} vs {final_epoch}\nWith Diagonal')
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("output_covariances_with_diagonal.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Visualize output covariances - WITHOUT diagonal (masked)
fig, axes = plt.subplots(1, len(compare_epochs), figsize=(20, 4))

for idx, epoch in enumerate(compare_epochs):
    cov = output_covs[epoch].cpu().numpy()
    
    # Mask diagonal
    cov_no_diag = cov.copy()
    np.fill_diagonal(cov_no_diag, np.nan)
    
    # Scale based on off-diagonal only
    vmax = np.nanmax(np.abs(cov_no_diag))
    
    im = axes[idx].imshow(cov_no_diag, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[idx].set_title(f'Epoch {epoch} vs {final_epoch}\nWithout Diagonal')
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("output_covariances_without_diagonal.png", dpi=300, bbox_inches='tight')
plt.show()


#%%
# Compute baseline confusion matrix (no ablation)
from sklearn.metrics import confusion_matrix

# Get final model predictions
final_model.eval()
with torch.no_grad():
    logits = final_model(test_data.x)
    preds = logits.argmax(dim=-1).cpu().numpy()
    
y_true = test_data.y.cpu().numpy()

# Baseline confusion matrix
cm_baseline = confusion_matrix(y_true, preds)

# Normalize to show error rates
cm_baseline_norm = cm_baseline.astype('float') / cm_baseline.sum(axis=1)[:, np.newaxis]

#%%
from einops import einsum
# Ablation function
def ablate_and_evaluate(model, digit_to_ablate=None, digit_pair=None):
    
    """
    Ablate weights and return confusion matrix.
    
    Args:
        model: The model to ablate
        digit_to_ablate: Single digit to ablate diagonal (e.g., 4)
        digit_pair: Tuple of digits to ablate off-diagonal interaction (e.g., (3, 4))
    
    Returns:
        Confusion matrix after ablation
    """
    # Create a copy to ablate
    ablated_model = Model(config)
    ablated_model.load_state_dict(model.state_dict())
    
    # Get interaction tensor
    l, r = ablated_model.w_lr[0].unbind(0)
    u = ablated_model.w_u
    B = einsum(u, l, r, "o h, h i, h j -> o i j")
    B = 0.5 * (B + B.transpose(1, 2))
    
    if digit_to_ablate is not None:
        # Ablate diagonal: zero out B[digit, :, :]
        B[digit_to_ablate, :, :] = 0
    
    if digit_pair is not None:
        # Ablate off-diagonal: zero out interaction between digit pair
        d1, d2 = digit_pair
        # This is trickier - we'd need to ablate the specific interaction
        # For now, let's zero both B[d1] and B[d2] contributions to each other
        # Actually, this requires decomposing back to L, R, U which is complex
        pass  # We'll focus on diagonal ablation first
    
    # Reconstruct weights (this is the hard part - we can't easily go B -> L, R, U)
    # Alternative: directly modify the weight matrices
    
    # For diagonal ablation, we can zero the unembedding row
    if digit_to_ablate is not None:
        ablated_model.head.weight.data[digit_to_ablate, :] = 0
    
    # Evaluate
    ablated_model.eval()
    with torch.no_grad():
        logits = ablated_model(test_data.x)
        preds = logits.argmax(dim=-1).cpu().numpy()
    
    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm_norm

#%%
# Test 1: Ablate digit 4 (strongest diagonal)
print("Ablating digit 4...")
cm_ablate_4 = ablate_and_evaluate(final_model, digit_to_ablate=4)

# Compare accuracy on digit 4
print(f"Baseline accuracy on digit 4: {cm_baseline_norm[4, 4]:.3f}")
print(f"After ablation accuracy on digit 4: {cm_ablate_4[4, 4]:.3f}")

#%%
# Visualize confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Baseline
im0 = axes[0].imshow(cm_baseline_norm, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Baseline Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
plt.colorbar(im0, ax=axes[0])

# Ablated
im1 = axes[1].imshow(cm_ablate_4, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('After Ablating Digit 4')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
plt.colorbar(im1, ax=axes[1])

# Difference
diff = cm_ablate_4 - cm_baseline_norm
im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[2].set_title('Difference (Ablated - Baseline)')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("ablation_digit4_confusion.png", dpi=300, bbox_inches='tight')
plt.show()


#%%
# Find weakest diagonal element
diagonal_values = {i: output_covs[99][i, i].item() for i in range(10)}
weakest_digit = min(diagonal_values, key=diagonal_values.get)

print(f"Diagonal values: {diagonal_values}")
print(f"Weakest diagonal: digit {weakest_digit} with value {diagonal_values[weakest_digit]:.3f}")

#%%
# Test 2: Ablate weakest diagonal digit
print(f"Ablating digit {weakest_digit}...")
cm_ablate_weak = ablate_and_evaluate(final_model, digit_to_ablate=weakest_digit)

# Compare accuracy
print(f"Baseline accuracy on digit {weakest_digit}: {cm_baseline_norm[weakest_digit, weakest_digit]:.3f}")
print(f"After ablation accuracy on digit {weakest_digit}: {cm_ablate_weak[weakest_digit, weakest_digit]:.3f}")

#%%
# Visualize comparison: strongest vs weakest
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Digit 4 (strongest)
axes[0, 0].imshow(cm_baseline_norm, cmap='Blues', vmin=0, vmax=1)
axes[0, 0].set_title('Baseline')
axes[0, 1].imshow(cm_ablate_4, cmap='Blues', vmin=0, vmax=1)
axes[0, 1].set_title(f'Ablate Digit 4 (strongest)')
diff_4 = cm_ablate_4 - cm_baseline_norm
axes[0, 2].imshow(diff_4, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0, 2].set_title('Difference')

# Row 2: Weakest digit
axes[1, 0].imshow(cm_baseline_norm, cmap='Blues', vmin=0, vmax=1)
axes[1, 0].set_title('Baseline')
axes[1, 1].imshow(cm_ablate_weak, cmap='Blues', vmin=0, vmax=1)
axes[1, 1].set_title(f'Ablate Digit {weakest_digit} (weakest)')
diff_weak = cm_ablate_weak - cm_baseline_norm
axes[1, 2].imshow(diff_weak, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[1, 2].set_title('Difference')

for ax in axes.flat:
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig("ablation_comparison_confusion.png", dpi=300, bbox_inches='tight')
plt.show()



#%%
# Test all diagonal ablations and check confusion patterns
digits_to_test = [0, 1, 2, 5, 6, 7, 8, 9]
expected_confusions = {
    0: [6, 9],      # 0 → 6 (maybe 9)
    1: [4, 7, 2],   # 1 → 4 (maybe 7, 2)
    2: [7, 1, 3],   # 2 → 8 (maybe 1, 3)
    5: [3,6],         # 5 → 4
    6: [0, 8],      # 6 → 0 or 8
    7: [1, 2],      # 7 → 1 (maybe 2)
    8: [6],         # 8 → 6
    9: [0, 4],      # 9 → 0 or 4
}

results = {}

for digit in digits_to_test:
    print(f"\nAblating digit {digit}...")
    cm_ablated = ablate_and_evaluate(final_model, digit_to_ablate=digit)
    
    # Get confusion row for ablated digit
    confusion_row = cm_ablated[digit, :]
    # Remove self (diagonal)
    confusion_row[digit] = 0
    
    # Find top confused classes
    top_confused = np.argsort(confusion_row)[::-1][:3]  # Top 3
    top_values = confusion_row[top_confused]
    
    results[digit] = {
        'top_confused': top_confused,
        'values': top_values,
        'expected': expected_confusions[digit]
    }
    
    print(f"  Expected confusion: {expected_confusions[digit]}")
    print(f"  Actual top confusions: {top_confused.tolist()} with values {top_values}")
    print(f"  Match: {top_confused[0] in expected_confusions[digit]}")

#%%
# Visualize all ablations in grid
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for idx, digit in enumerate(digits_to_test):
    cm_ablated = ablate_and_evaluate(final_model, digit_to_ablate=digit)
    diff = cm_ablated - cm_baseline_norm
    
    im = axes[idx].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[idx].set_title(f'Ablate Digit {digit}\nExpected: {expected_confusions[digit]}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
    
    # Highlight expected confusion cells
    for confused_digit in expected_confusions[digit]:
        axes[idx].add_patch(plt.Rectangle((confused_digit-0.5, digit-0.5), 1, 1, 
                                         fill=False, edgecolor='green', linewidth=3))
    
    plt.colorbar(im, ax=axes[idx])

# Hide last empty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig("all_diagonal_ablations.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
def ablate_and_evaluate_with_B(model, digit_to_ablate=None):
    """
    Ablate by modifying B directly and using quadratic form for inference.
    """
    # Build B tensor
    l, r = model.w_lr[0].unbind(0)
    u = model.w_u
    B = einsum(u, l, r, "o h, h i, h j -> o i j")
    B = 0.5 * (B + B.transpose(1, 2))
    
    # Ablate
    if digit_to_ablate is not None:
        B[digit_to_ablate, :, :] = 0
    
    # Inference using B
    model.eval()
    with torch.no_grad():
        x_embedded = model.embed(test_data.x.flatten(start_dim=1))  # [batch, d_hidden]
        
        # Compute logits: for each class o, compute x^T B[o] x
        logits = einsum(x_embedded, B, x_embedded, 
                       "batch i, o i j, batch j -> batch o")
        
        preds = logits.argmax(dim=-1).cpu().numpy()
    
    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm_norm



# Rerun digit 8 ablation with B-based method
print("Ablating digit 8 with B-based method...")
cm_ablate_8_B = ablate_and_evaluate_with_B(final_model, digit_to_ablate=8)

# Compare with original method
print(f"Original method - accuracy on digit 8: {cm_ablate_4[8, 8]:.3f}")
print(f"B-based method - accuracy on digit 8: {cm_ablate_8_B[8, 8]:.3f}")


# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Baseline
axes[0].imshow(cm_baseline_norm, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Baseline')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# B-based ablation
axes[1].imshow(cm_ablate_8_B, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Ablate Digit 8 (B-based)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

# Difference
diff = cm_ablate_8_B - cm_baseline_norm
axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[2].set_title('Difference (B-based - Baseline)')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')

# Highlight expected confusion (digit 6, from positive covariance)
axes[2].add_patch(plt.Rectangle((6-0.5, 8-0.5), 1, 1, 
                                fill=False, edgecolor='green', linewidth=3))

plt.tight_layout()
plt.savefig("ablation_digit8_B_method.png", dpi=300, bbox_inches='tight')
plt.show()


# Check top confusions
confusion_row = cm_ablate_8_B[8, :]
confusion_row[8] = 0  # Remove self
top_confused = np.argsort(confusion_row)[::-1][:3]
print(f"\nDigit 8 confuses with: {top_confused} (expected: 6)")
print(f"Confusion values: {confusion_row[top_confused]}")


#%%
# Analyze which hidden dimensions drive covariance
def analyze_feature_overlap(model1, model2, digit1, digit2, layer_idx=0):
    """
    Show which hidden dimensions contribute to covariance between two digits.
    """
    l1, r1 = model1.w_lr[layer_idx].unbind(0)
    l2, r2 = model2.w_lr[layer_idx].unbind(0)
    u1 = model1.w_u
    u2 = model2.w_u
    
    # Build B matrices
    B1 = einsum(u1, l1, r1, "o h, h i, h j -> o i j")
    B1 = 0.5 * (B1 + B1.transpose(1, 2))
    
    B2 = einsum(u2, l2, r2, "o h, h i, h j -> o i j")
    B2 = 0.5 * (B2 + B2.transpose(1, 2))
    
    # Get interaction matrices for specific digits
    B1_d = B1[digit1]  # [256, 256]
    B2_d = B2[digit2]  # [256, 256]
    
    # Contribution of each hidden dimension pair (i,j) to covariance
    contributions = B1_d * B2_d  # [256, 256]
    
    # Sum over j for each i to see per-dimension contribution
    per_dim_i = contributions.sum(dim=1)  # [256]
    per_dim_j = contributions.sum(dim=0)  # [256]
    
    return {
        'contributions_matrix': contributions,
        'per_dim_i': per_dim_i,
        'per_dim_j': per_dim_j,
        'total_covariance': contributions.sum().item()
    }

#%%
# Compare positive vs negative covariance pairs
# Positive pair: 1 and 4 (share features)
pos_analysis = analyze_feature_overlap(final_model, final_model, 1, 4)

# Negative pair: 3 and 4 (oppose features)
neg_analysis = analyze_feature_overlap(final_model, final_model, 3, 4)

#%%
# Visualize feature contributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Positive covariance (1 vs 4)
axes[0, 0].imshow(pos_analysis['contributions_matrix'].cpu().numpy(), cmap='RdBu_r')
axes[0, 0].set_title('1 vs 4: Contribution Matrix\n(Positive Covariance)')
axes[0, 0].set_xlabel('Hidden Dim j')
axes[0, 0].set_ylabel('Hidden Dim i')

axes[0, 1].bar(range(256), pos_analysis['per_dim_i'].cpu().numpy())
axes[0, 1].set_title('Per-dimension contribution (summed over j)')
axes[0, 1].set_xlabel('Hidden Dimension i')
axes[0, 1].set_ylabel('Contribution')
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

axes[0, 2].hist(pos_analysis['contributions_matrix'].flatten().cpu().numpy(), bins=501)
axes[0, 2].set_title('Distribution of contributions')
axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)

# Row 2: Negative covariance (3 vs 4)
axes[1, 0].imshow(neg_analysis['contributions_matrix'].cpu().numpy(), cmap='RdBu_r')
axes[1, 0].set_title('3 vs 4: Contribution Matrix\n(Negative Covariance)')
axes[1, 0].set_xlabel('Hidden Dim j')
axes[1, 0].set_ylabel('Hidden Dim i')

axes[1, 1].bar(range(256), neg_analysis['per_dim_i'].cpu().numpy())
axes[1, 1].set_title('Per-dimension contribution (summed over j)')
axes[1, 1].set_xlabel('Hidden Dimension i')
axes[1, 1].set_ylabel('Contribution')
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

axes[1, 2].hist(neg_analysis['contributions_matrix'].flatten().cpu().numpy(), bins=501)
axes[1, 2].set_title('Distribution of contributions')
axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig("feature_overlap_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"0 vs 6 (positive): Total covariance = {pos_analysis['total_covariance']:.3f}")
print(f"3 vs 4 (negative): Total covariance = {neg_analysis['total_covariance']:.3f}")



#%%
# Orthogonalization functions
def orthogonalize_digits(model, digit1, digit2, layer_idx=0):
    """
    Make B[digit1] and B[digit2] orthogonal by removing their shared component.
    Uses Gram-Schmidt to orthogonalize digit2 w.r.t. digit1.
    """
    # Build B
    l, r = model.w_lr[layer_idx].unbind(0)
    u = model.w_u
    B = einsum(u, l, r, "o h, h i, h j -> o i j")
    B = 0.5 * (B + B.transpose(1, 2))
    
    # Flatten B matrices for the two digits
    B1_flat = B[digit1].flatten()  # [256*256]
    B2_flat = B[digit2].flatten()
    
    # Compute projection of B2 onto B1
    proj_coeff = (B1_flat @ B2_flat) / (B1_flat @ B1_flat)
    projection = proj_coeff * B1_flat
    
    # Remove projection from B2 (Gram-Schmidt)
    B2_orthog = B2_flat - projection
    
    # Reshape back
    B[digit2] = B2_orthog.reshape(256, 256)
    
    return B

def evaluate_with_modified_B(B, embedding_weight):
    """
    Evaluate model using modified B tensor directly.
    """
    with torch.no_grad():
        # Embed test data
        x_embedded = test_data.x.flatten(start_dim=1) @ embedding_weight.T
        
        # Compute logits using quadratic form
        logits = einsum(x_embedded, B, x_embedded, 
                       "batch i, o i j, batch j -> batch o")
        
        preds = logits.argmax(dim=-1).cpu().numpy()
    
    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm_norm

#%%
# Test orthogonalization on three cases

# Case 1: Strong positive covariance (1 vs 4)
print("Case 1: Orthogonalizing digits 1 and 4 (strong positive covariance)")
print(f"Original covariance (1,4): {output_covs[99][1,4].item():.4f}")

B_orthog_1_4 = orthogonalize_digits(final_model, 1, 4)
B1_flat = B_orthog_1_4[1].flatten()
B4_flat = B_orthog_1_4[4].flatten()
print(f"After orthogonalization covariance (1,4): {(B1_flat @ B4_flat).item():.4f}")

cm_orthog_1_4 = evaluate_with_modified_B(B_orthog_1_4, final_model.embed.weight.data)

print(f"Baseline accuracy on 1: {cm_baseline_norm[1,1]:.4f}, After: {cm_orthog_1_4[1,1]:.4f}")
print(f"Baseline accuracy on 4: {cm_baseline_norm[4,4]:.4f}, After: {cm_orthog_1_4[4,4]:.4f}")

#%%
# Case 2: Strong negative covariance (3 vs 4)
print("\nCase 2: Orthogonalizing digits 3 and 4 (strong negative covariance)")
print(f"Original covariance (3,4): {output_covs[99][3,4].item():.4f}")

B_orthog_3_4 = orthogonalize_digits(final_model, 3, 4)
B3_flat = B_orthog_3_4[3].flatten()
B4_flat_2 = B_orthog_3_4[4].flatten()
print(f"After orthogonalization covariance (3,4): {(B3_flat @ B4_flat_2).item():.4f}")

cm_orthog_3_4 = evaluate_with_modified_B(B_orthog_3_4, final_model.embed.weight.data)

print(f"Baseline accuracy on 3: {cm_baseline_norm[3,3]:.4f}, After: {cm_orthog_3_4[3,3]:.4f}")
print(f"Baseline accuracy on 4: {cm_baseline_norm[4,4]:.4f}, After: {cm_orthog_3_4[4,4]:.4f}")

#%%
# Case 3: Near-zero covariance (0 vs 8)
print("\nCase 3: Orthogonalizing digits 0 and 8 (near-zero covariance)")
print(f"Original covariance (0,8): {output_covs[99][0,8].item():.4f}")

B_orthog_0_8 = orthogonalize_digits(final_model, 0, 8)
B0_flat = B_orthog_0_8[0].flatten()
B8_flat_2 = B_orthog_0_8[8].flatten()
print(f"After orthogonalization covariance (0,8): {(B0_flat @ B8_flat_2).item():.4f}")

cm_orthog_0_8 = evaluate_with_modified_B(B_orthog_0_8, final_model.embed.weight.data)

print(f"Baseline accuracy on 0: {cm_baseline_norm[0,0]:.4f}, After: {cm_orthog_0_8[0,0]:.4f}")
print(f"Baseline accuracy on 8: {cm_baseline_norm[8,8]:.4f}, After: {cm_orthog_0_8[8,8]:.4f}")

#%%
# Visualize all three cases
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

cases = [
    (cm_orthog_1_4, '1 & 4 (Strong Positive)', (1, 4)),
    (cm_orthog_3_4, '3 & 4 (Strong Negative)', (3, 4)),
    (cm_orthog_0_8, '0 & 8 (Near Zero)', (0, 8))
]

for row, (cm_orthog, title, digits) in enumerate(cases):
    diff = cm_orthog - cm_baseline_norm
    
    axes[row, 0].imshow(cm_baseline_norm, cmap='Blues', vmin=0, vmax=1)
    axes[row, 0].set_title('Baseline')
    axes[row, 0].set_ylabel(title, fontsize=12, fontweight='bold')
    
    axes[row, 1].imshow(cm_orthog, cmap='Blues', vmin=0, vmax=1)
    axes[row, 1].set_title(f'After Orthogonalizing {title}')
    
    axes[row, 2].imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    axes[row, 2].set_title('Difference (Orthog - Baseline)')
    
    # Highlight affected digits
    for digit in digits:
        for col in range(3):
            axes[row, col].axhline(y=digit, color='yellow', linestyle='--', linewidth=1, alpha=0.5)
            axes[row, col].axvline(x=digit, color='yellow', linestyle='--', linewidth=1, alpha=0.5)

for ax in axes.flat:
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig("orthogonalization_three_cases.png", dpi=300, bbox_inches='tight')
plt.show()





#%%
# Compute pairwise classification accuracy for digit pairs

# Define digit pairs to test
digit_pairs = [
    (1, 2, "Similar: 1 vs 2"),
    (5, 6, "Similar: 5 vs 6"),
    (3, 9, "Similar: 3 vs 9"),
    (0, 8, "Similar: 0 vs 8"),
    (1, 4, "Different: 1 vs 4"),
    (7, 8, "Different: 7 vs 8"),
]

# Track accuracy and covariance over training
pair_results = {label: {'epochs': [], 'accuracy': [], 'covariance': []} 
                for _, _, label in digit_pairs}

#%%
# Compute metrics for all checkpoints
from functions.tn_covariance import interaction_output_covariance

for cp in tqdm(checkpoints, desc="Computing pairwise metrics"):
    epoch = cp['epoch']
    model_t = Model(config)
    model_t.load_state_dict(cp['state_dict'])
    
    # Compute output covariance for this checkpoint vs final
    output_cov = interaction_output_covariance(model_t, final_model, layer_idx=0).cpu().numpy()
    
    # For each digit pair
    for d1, d2, label in digit_pairs:
        # Compute pairwise accuracy inline
        mask = (test_data.y == d1) | (test_data.y == d2)
        x_pair = test_data.x[mask]
        y_pair = test_data.y[mask]
        
        model_t.eval()
        with torch.no_grad():
            logits = model_t(x_pair)
            preds = logits.argmax(dim=-1)
        
        acc = (preds == y_pair).float().mean().item()
        cov = output_cov[d1, d2]
        
        pair_results[label]['epochs'].append(epoch)
        pair_results[label]['accuracy'].append(acc)
        pair_results[label]['covariance'].append(cov)

#%%
# Plot accuracy vs covariance for each pair
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (_, _, label) in enumerate(digit_pairs):
    ax = axes[idx]
    results = pair_results[label]
    
    scatter = ax.scatter(results['covariance'], results['accuracy'], 
                        c=results['epochs'], cmap='viridis', s=100, alpha=0.7)
    
    ax.set_xlabel('Output Covariance')
    ax.set_ylabel('Pairwise Accuracy')
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Epoch')

plt.tight_layout()
plt.savefig("pairwise_accuracy_vs_covariance.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot accuracy vs covariance for each pair (same scales)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Find global min/max for consistent scales
all_cov = [cov for label in pair_results for cov in pair_results[label]['covariance']]
all_acc = [acc for label in pair_results for acc in pair_results[label]['accuracy']]
cov_min, cov_max = min(all_cov), max(all_cov)
acc_min, acc_max = min(all_acc), max(all_acc)

for idx, (_, _, label) in enumerate(digit_pairs):
    ax = axes[idx]
    results = pair_results[label]
    
    scatter = ax.scatter(results['covariance'], results['accuracy'], 
                        c=results['epochs'], cmap='viridis', s=100, alpha=0.7)
    
    ax.set_xlabel('Output Covariance')
    ax.set_ylabel('Pairwise Accuracy')
    ax.set_title(label)
    ax.set_xlim(cov_min, cov_max)
    ax.set_ylim(acc_min, acc_max)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Epoch')

plt.tight_layout()
plt.savefig("pairwise_accuracy_vs_covariance.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
