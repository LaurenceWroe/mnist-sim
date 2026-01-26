#%%
# Investigate why initial covariance is so similar
print("\n=== Investigating Initial Covariance Similarity ===")

# Look at batch 0 covariances
init_covs = {}
for seed in seeds:
    model_init = Model(models_data[seed]['config'])
    model_init.load_state_dict(models_data[seed]['checkpoints_sparse'][0]['state_dict'])
    cov_init = interaction_output_covariance(model_init, model_init, layer_idx=0).cpu().numpy()
    init_covs[seed] = cov_init

# Visualize initial covariances
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, seed in enumerate(seeds):
    cov_viz = init_covs[seed].copy()
    # np.fill_diagonal(cov_viz, np.nan)
    # vmax = np.nanmax(np.abs(cov_viz))
    
    
    im = axes[idx].imshow(cov_viz, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[idx].set_title(f'Seed {seed} - Batch 0\n(Random Init)')
    axes[idx].set_xlabel('Output Class')
    axes[idx].set_ylabel('Output Class')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig(figures_dir / "initial_covariances_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Compute statistics
print("\nInitial covariance statistics:")
for seed in seeds:
    cov = init_covs[seed]
    print(f"\nSeed {seed}:")
    print(f"  Diagonal mean: {np.diag(cov).mean():.4f} ± {np.diag(cov).std():.4f}")
    print(f"  Off-diagonal mean: {cov[~np.eye(10, dtype=bool)].mean():.4f}")
    print(f"  Off-diagonal std: {cov[~np.eye(10, dtype=bool)].std():.4f}")
    print(f"  Frobenius norm: {np.linalg.norm(cov):.4f}")

# Compare to random matrices
print("\n=== Comparison to Random Matrices ===")
np.random.seed(999)
random_covs = []
for i in range(3):
    # Generate random symmetric matrix
    A = np.random.randn(10, 10)
    A = (A + A.T) / 2  # Make symmetric
    random_covs.append(A)

# Compute similarity between random matrices
for i in range(3):
    for j in range(i+1, 3):
        sim = covariance_similarity(random_covs[i], random_covs[j])
        print(f"Random matrix {i} vs {j}: {sim:.4f}")

# Compare to our initial covariances
print("\nOur initial covariances similarity:")
for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        sim = covariance_similarity(init_covs[seed1], init_covs[seed2])
        print(f"Seed {seed1} vs {seed2}: {sim:.4f}")


# %%


#%%
# Test: Does the diagonal dominate the similarity metric?
print("\n=== Testing Diagonal Dominance ===")

# Compare similarity WITH vs WITHOUT diagonal
for seed1 in seeds:
    for seed2 in seeds:
        if seed1 < seed2:
            # Full covariance similarity
            sim_full = covariance_similarity(init_covs[seed1], init_covs[seed2])
            
            # Remove diagonal
            cov1_no_diag = init_covs[seed1].copy()
            cov2_no_diag = init_covs[seed2].copy()
            np.fill_diagonal(cov1_no_diag, 0)
            np.fill_diagonal(cov2_no_diag, 0)
            
            sim_no_diag = covariance_similarity(cov1_no_diag, cov2_no_diag)
            
            print(f"Seed {seed1} vs {seed2}:")
            print(f"  With diagonal:    {sim_full:.4f}")
            print(f"  Without diagonal: {sim_no_diag:.4f}")
            print(f"  Difference:       {sim_full - sim_no_diag:.4f}\n")

#%%
# Compute the contribution of diagonal vs off-diagonal to the total norm
print("\n=== Diagonal vs Off-diagonal Contribution ===")

for seed in seeds:
    cov = init_covs[seed]
    
    # Total Frobenius norm squared
    total_norm_sq = np.sum(cov**2)
    
    # Diagonal contribution
    diag_norm_sq = np.sum(np.diag(cov)**2)
    
    # Off-diagonal contribution
    off_diag_norm_sq = total_norm_sq - diag_norm_sq
    
    print(f"Seed {seed}:")
    print(f"  Diagonal contribution:     {diag_norm_sq / total_norm_sq * 100:.1f}%")
    print(f"  Off-diagonal contribution: {off_diag_norm_sq / total_norm_sq * 100:.1f}%\n")

# %%
