import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = K.shape[-1]
    values = (Q @ K.T) / np.sqrt(d_k)

    exp_s = np.exp(values - np.max(values, axis=-1, keepdims=True))
    exp_f = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    return exp_f @ V
