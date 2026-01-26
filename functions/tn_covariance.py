import torch
from einops import einsum

def interaction_output_covariance(model1, model2, layer_idx=0):
    """
    Compute covariance between output classes by contracting interaction matrices
    over hidden dimensions.
    
    Args:
        model1: First model
        model2: Second model
        layer_idx: Which layer
    
    Returns:
        Covariance matrix [d_output, d_output] showing how output classes
        co-vary between the two models
    """
    # Form interaction tensors
    l1, r1 = model1.w_lr[layer_idx].unbind(0)
    l2, r2 = model2.w_lr[layer_idx].unbind(0)
    u1 = model1.w_u
    u2 = model2.w_u
    
    # B1, B2 shape: [d_output, d_hidden, d_hidden]
    B1 = einsum(u1, l1, r1, "o h, h i, h j -> o i j")
    B1 = 0.5 * (B1 + B1.transpose(1, 2))
    
    B2 = einsum(u2, l2, r2, "o h, h i, h j -> o i j")
    B2 = 0.5 * (B2 + B2.transpose(1, 2))
    
    # Contract over hidden dimensions: sum over i,j
    # Result: [d_output, d_output]
    output_cov = einsum(B1, B2, "o1 i j, o2 i j -> o1 o2")
    
    return output_cov

def interaction_latent_covariance(model1, model2, layer_idx=0):
    """
    Compute latent covariance by contracting L and R matrices directly,
    without including the unembedding layer.
    
    Args:
        model1: First model
        model2: Second model
        layer_idx: Which layer
    
    Returns:
        Covariance matrix [d_hidden, d_hidden] showing how latent dimensions
        co-vary between the two models in the bilinear interaction
    """
    # Extract L and R matrices only
    l1, r1 = model1.w_lr[layer_idx].unbind(0)  # [d_hidden, d_hidden]
    l2, r2 = model2.w_lr[layer_idx].unbind(0)
    
    # Contract L1⊙R1 with L2⊙R2 over one hidden dimension
    # This gives covariance in the latent interaction space
    latent_cov = einsum(l1, r1, l2, r2, "h1 i, h1 j, h2 i, h2 j -> h1 h2")
    
    return latent_cov