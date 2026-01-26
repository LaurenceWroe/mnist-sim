"""
Tensor similarity computation for tracking training dynamics.
"""
import torch
from einops import einsum


def tensor_similarity_single_layer(model1, model2, layer_idx=0):
    """
    Compute symmetric tensor similarity between two single-layer bilinear models.
    
    Args:
        model1: First model (or checkpoint)
        model2: Second model (or checkpoint)
        layer_idx: Which layer to compare (default 0 for single-layer models)
    
    Returns:
        Scalar similarity value in [0, 1]
    """
    # Extract L and R matrices for the specified layer
    l1, r1 = model1.w_lr[layer_idx].unbind(0)  # [d_hidden, d_hidden] each
    l2, r2 = model2.w_lr[layer_idx].unbind(0)
    
    # Extract unembedding (head) weights
    u1 = model1.w_u  # [d_output, d_hidden]
    u2 = model2.w_u
    
    # Compute inner product components
    # L1 · L2 and R1 · R2
    ll = einsum(l1, l2, "hid1 i, hid2 i -> hid1 hid2")
    rr = einsum(r1, r2, "hid1 i, hid2 i -> hid1 hid2")
    
    # L1 · R2 and R1 · L2 (for symmetry)
    lr = einsum(l1, r2, "hid1 i, hid2 i -> hid1 hid2")
    rl = einsum(r1, l2, "hid1 i, hid2 i -> hid1 hid2")
    
    # Symmetric core: 0.5 * (L1·L2 * R1·R2 + L1·R2 * R1·L2)
    core = 0.5 * ((ll * rr) + (lr * rl))
    
    # Contract with unembedding matrices
    dd = einsum(u1, u2, "o hid1, o hid2 -> hid1 hid2")
    hid = einsum(core, dd, "hid1 hid2, hid1 hid3 -> hid2 hid3")
    inner_product = torch.trace(hid)
    
    # Compute norms
    norm1 = compute_tensor_norm(model1, layer_idx)
    norm2 = compute_tensor_norm(model2, layer_idx)
    
    # Return cosine similarity
    similarity = inner_product / (norm1 * norm2)
    return similarity.item()


def tensor_similarity_with_embedding(model1, model2, layer_idx=0):
    """
    Compute tensor similarity including the embedding layer.
    Full path: input → embed → bilinear → head
    
    Args:
        model1: First model
        model2: Second model  
        layer_idx: Which layer to compare
    
    Returns:
        Scalar similarity value in [0, 1]
    """
    # Extract all weights
    e1 = model1.embed.weight  # [d_hidden, d_input]
    e2 = model2.embed.weight
    
    l1, r1 = model1.w_lr[layer_idx].unbind(0)  # [d_hidden, d_hidden]
    l2, r2 = model2.w_lr[layer_idx].unbind(0)
    
    u1 = model1.w_u  # [d_output, d_hidden]
    u2 = model2.w_u
    
    # The full tensor for output o, input positions (i,j):
    # T[o, i, j] = sum_h1,h2,h3 U[o,h3] * L[h3,h2] * R[h3,h1] * E[h2,i] * E[h1,j]
    # Or with symmetrization:
    # T[o, i, j] = sum_h1,h2,h3 U[o,h3] * 0.5*(L[h3,h2]*R[h3,h1] + L[h3,h1]*R[h3,h2]) * E[h2,i] * E[h1,j]
    
    # Build interaction tensor with embedding
    # T1[o, i, j] for model 1
    # First: E @ L.T and E @ R.T give [d_input, d_hidden]
    el1 = einsum(e1, l1, "h i, h j -> i j")  # E.T @ L: [d_input, d_hidden]
    er1 = einsum(e1, r1, "h i, h j -> i j")  # E.T @ R: [d_input, d_hidden]
    
    el2 = einsum(e2, l2, "h i, h j -> i j")
    er2 = einsum(e2, r2, "h i, h j -> i j")
    
    # Inner products for similarity computation
    # <E1@L1, E2@L2> and <E1@R1, E2@R2>
    el_el = einsum(el1, el2, "i h1, i h2 -> h1 h2")
    er_er = einsum(er1, er2, "i h1, i h2 -> h1 h2")
    
    # Cross terms for symmetry
    el_er = einsum(el1, er2, "i h1, i h2 -> h1 h2")
    er_el = einsum(er1, el2, "i h1, i h2 -> h1 h2")
    
    # Symmetric core
    core = 0.5 * ((el_el * er_er) + (el_er * er_el))
    
    # Contract with unembedding
    uu = einsum(u1, u2, "o h1, o h2 -> h1 h2")
    hid = einsum(core, uu, "h1 h2, h1 h3 -> h2 h3")
    inner_product = torch.trace(hid)
    
    # Compute norms
    norm1 = compute_tensor_norm_with_embedding(model1, layer_idx)
    norm2 = compute_tensor_norm_with_embedding(model2, layer_idx)
    
    similarity = inner_product / (norm1 * norm2)
    return similarity.item()


def compute_tensor_norm_with_embedding(model, layer_idx=0):
    """
    Compute tensor norm including embedding layer.
    """
    e = model.embed.weight
    l, r = model.w_lr[layer_idx].unbind(0)
    u = model.w_u
    
    # E @ L and E @ R
    el = einsum(e, l, "h i, h j -> i j")
    er = einsum(e, r, "h i, h j -> i j")
    
    # Self inner products
    el_el = einsum(el, el, "i h1, i h2 -> h1 h2")
    er_er = einsum(er, er, "i h1, i h2 -> h1 h2")
    el_er = einsum(el, er, "i h1, i h2 -> h1 h2")
    
    core = 0.5 * ((el_el * er_er) + (el_er * el_er.T))
    
    uu = einsum(u, u, "o h1, o h2 -> h1 h2")
    hid = einsum(core, uu, "h1 h2, h1 h3 -> h2 h3")
    
    return torch.sqrt(torch.trace(hid))

def interaction_matrix_similarity(model1, model2, layer_idx=0):
    # Extract weights
    l1, r1 = model1.w_lr[layer_idx].unbind(0)
    l2, r2 = model2.w_lr[layer_idx].unbind(0)
    u1 = model1.w_u
    u2 = model2.w_u
    
    # Form interaction tensors with symmetrization: B = U @ (L ⊗ R)
    # Then symmetrize: 0.5 * (B + B.transpose)
    B1 = einsum(u1, l1, r1, "o h, h i, h j -> o i j")
    B1 = 0.5 * (B1 + B1.transpose(1, 2))  # Symmetrize over i,j
    
    B2 = einsum(u2, l2, r2, "o h, h i, h j -> o i j")
    B2 = 0.5 * (B2 + B2.transpose(1, 2))

    # Cosine similarity using einsum (no flattening needed)
    dot_product = einsum(B1, B2, "o i j, o i j ->")  # Sum over all indices
    norm1 = torch.sqrt(einsum(B1, B1, "o i j, o i j ->"))
    norm2 = torch.sqrt(einsum(B2, B2, "o i j, o i j ->"))
    
    # # Flatten to vectors
    # B1_flat = B1.flatten()
    # B2_flat = B2.flatten()
    
    # # Cosine similarity
    # dot_product = (B1_flat * B2_flat).sum()
    # norm1 = torch.sqrt((B1_flat ** 2).sum())
    # norm2 = torch.sqrt((B2_flat ** 2).sum())
    
    return (dot_product / (norm1 * norm2)).item()

def interaction_matrix_similarity_new(model1, model2, layer_idx=0, include_embedding=True):
    """
    Compute similarity between interaction matrices.
    
    Args:
        model1: First model
        model2: Second model
        layer_idx: Which layer (default 0)
        include_embedding: Whether to include embedding layer (default True)
    
    Returns:
        Scalar similarity value
    """
    if include_embedding:
        # Extract all weights including embedding
        e1 = model1.embed.weight  # [d_hidden, d_input]
        e2 = model2.embed.weight
        
        l1, r1 = model1.w_lr[layer_idx].unbind(0)
        l2, r2 = model2.w_lr[layer_idx].unbind(0)
        u1 = model1.w_u
        u2 = model2.w_u
        
        # Contract embedding with L and R: [d_input, d_hidden]
        # E is [d_hidden, d_input], so E.T @ L gives [d_input, d_hidden]
        el1 = einsum(e1, l1, "h1 i, h1 h2 -> i h2")  # [d_input, d_hidden]
        er1 = einsum(e1, r1, "h1 i, h1 h2 -> i h2")  # [d_input, d_hidden]
        
        el2 = einsum(e2, l2, "h1 i, h1 h2 -> i h2")
        er2 = einsum(e2, r2, "h1 i, h1 h2 -> i h2")
        
        # Form full interaction tensors: T[o, i, j] where i,j are input positions
        # T = U @ (EL ⊗ ER) with symmetrization
        T1 = einsum(u1, el1, er1, "o h, i h, j h -> o i j")
        T1 = 0.5 * (T1 + T1.transpose(1, 2))  # Symmetrize over i,j
        
        T2 = einsum(u2, el2, er2, "o h, i h, j h -> o i j")
        T2 = 0.5 * (T2 + T2.transpose(1, 2))
        
    else:
        # Original version without embedding
        l1, r1 = model1.w_lr[layer_idx].unbind(0)
        l2, r2 = model2.w_lr[layer_idx].unbind(0)
        u1 = model1.w_u
        u2 = model2.w_u
        
        # Form interaction tensors: B[o, i, j] where i,j are hidden dimensions
        T1 = einsum(u1, l1, r1, "o h, h i, h j -> o i j")
        T1 = 0.5 * (T1 + T1.transpose(1, 2))
        
        T2 = einsum(u2, l2, r2, "o h, h i, h j -> o i j")
        T2 = 0.5 * (T2 + T2.transpose(1, 2))
    
    # Cosine similarity
    dot_product = einsum(T1, T2, "o i j, o i j ->")
    norm1 = torch.sqrt(einsum(T1, T1, "o i j, o i j ->"))
    norm2 = torch.sqrt(einsum(T2, T2, "o i j, o i j ->"))
    
    return (dot_product / (norm1 * norm2)).item()

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

def compute_tensor_norm(model, layer_idx=0):
    """
    Compute the tensor norm (sqrt of inner product with itself).
    
    Args:
        model: The model
        layer_idx: Which layer (default 0)
    
    Returns:
        Scalar norm value
    """
    l, r = model.w_lr[layer_idx].unbind(0)
    u = model.w_u
    
    # Self inner product
    ll = einsum(l, l, "hid1 i, hid2 i -> hid1 hid2")
    rr = einsum(r, r, "hid1 i, hid2 i -> hid1 hid2")
    lr = einsum(l, r, "hid1 i, hid2 i -> hid1 hid2")
    
    # For self-inner product: lr and rl are transposes
    # So we need lr * lr.T (which equals lr * rl when model1==model2)
    core = 0.5 * ((ll * rr) + (lr * lr.T))
    
    dd = einsum(u, u, "o hid1, o hid2 -> hid1 hid2")
    hid = einsum(core, dd, "hid1 hid2, hid1 hid3 -> hid2 hid3")
    
    return torch.sqrt(torch.trace(hid))


def compute_similarity_to_init(model, init_state_dict, layer_idx=0):
    """
    Compute tensor similarity between current model and its initialization.
    
    Args:
        model: Current model
        init_state_dict: State dict of the model at initialization
        layer_idx: Which layer to compare
    
    Returns:
        Scalar similarity value
    """
    # Create a temporary model from init state
    init_model = type(model)(model.config)
    init_model.load_state_dict(init_state_dict)
    
    return tensor_similarity_single_layer(model, init_model, layer_idx)