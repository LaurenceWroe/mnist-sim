"""
Tensor similarity computation for tracking training dynamics.
"""
import torch
from einops import einsum

def get_interaction_matrix(model):
    """Extract the full interaction matrix B[cls, in1, in2]."""
    l, r = model.w_lr[0].unbind()
    B = einsum(model.w_u, l, r, "c o, o i, o j -> c i j")
    B = 0.5 * (B + B.transpose(1, 2))
    return B

def get_interaction_tensor_with_embedding(model):
    """Extract the full interaction tensor T[o, i, j] including embedding."""
    e = model.embed.weight  # [d_hidden, d_input]
    l, r = model.w_lr[0].unbind()  # [d_hidden, d_hidden]
    u = model.w_u  # [d_output, d_hidden]

    # Contract E with L and R, keeping hidden dimension
    # el[i, h] = sum over h1: E[h1, i] * L[h1, h]
    el = einsum(e, l, "h1 i, h1 h -> i h")  # [d_input, d_hidden]
    er = einsum(e, r, "h1 i, h1 h -> i h")  # [d_input, d_hidden]

    # Compute full interaction tensor T[o, i, j]
    # Contract over hidden dimension
    T = einsum(u, el, er, "o h, i h, j h -> o i j")
    T = 0.5 * (T + T.transpose(1, 2))  # Symmetrize
    
    return T

def int_tensor_similarity(T1, T2):
    # Cosine similarity
    dot_product = einsum(T1, T2, "o i j, o i j ->")
    norm1 = torch.sqrt(einsum(T1, T1, "o i j, o i j ->"))
    norm2 = torch.sqrt(einsum(T2, T2, "o i j, o i j ->"))
    
    return (dot_product / (norm1 * norm2)).item()
