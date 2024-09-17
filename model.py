import jax.numpy as jnp
import jax

# ---------- Utils ----------
def init_param(key, shape: tuple):
    # He initialization
    n_prev = shape[1] if len(shape) > 1 else shape[0]
    std = jnp.sqrt(2 / n_prev)
    gaussian_scaled = jax.random.normal(key, shape) * std
    return gaussian_scaled

def softmax(x):
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)

def dropout(key, z: jnp.ndarray, drop_rate: float, train: bool = True):
    if not train:
        return z
    mask = jax.random.bernoulli(key, 1 - drop_rate, z.shape) / (1 - drop_rate)
    return z * mask

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta

# ---------- Transformers Utils ----------
def embedding(x, w_e):
    return w_e[x]

def positional_encoding(x, w_p):
    seq_len = x.shape[1]
    return x + w_p[:seq_len]  # match positional embedding to input sequence length

def self_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
    attention = softmax(attn_logits)
    return jnp.matmul(attention, v)

def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads):
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    q = jnp.matmul(x, w_q).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k = jnp.matmul(x, w_k).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    v = jnp.matmul(x, w_v).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    attn_output = self_attention(q, k, v)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return jnp.matmul(attn_output, w_o)

def feed_forward(x, w1, b1, w2, b2):
    return jnp.matmul(jax.nn.relu(jnp.matmul(x, w1) + b1), w2) + b2

def transformer_block(x, params, num_heads, dropout_rate, train, key):
    key, attn_key, ff_key = jax.random.split(key, 3)
    
    attn_output = multi_head_attention(x, params['w_q'], params['w_k'], params['w_v'], params['w_o'], num_heads)
    attn_output = dropout(attn_key, attn_output, dropout_rate, train)
    x = layer_norm(x + attn_output, params['ln1_gamma'], params['ln1_beta'])
    
    ff_output = feed_forward(x, params['w1'], params['b1'], params['w2'], params['b2'])
    ff_output = dropout(ff_key, ff_output, dropout_rate, train)
    x = layer_norm(x + ff_output, params['ln2_gamma'], params['ln2_beta'])
    
    return x

def forward(x, params: dict, num_layers: int, num_heads: int, dropout_rate: float, train: bool = True):
    key = jax.random.PRNGKey(0)
    key, dropout_key = jax.random.split(key)

    x = embedding(x, params['w_embedding'])
    x = positional_encoding(x, params['w_positional'])
    
    for i in range(num_layers):
        key, block_key = jax.random.split(key)
        x = transformer_block(x, params[f'block_{i}'], num_heads, dropout_rate, train, block_key)
    
    x = layer_norm(x, params['ln_final_gamma'], params['ln_final_beta'])
    
    logits = jnp.matmul(x, params['w_lm_head'])
    
    return logits

if __name__ == "__main__":
    vocab_size = 1000
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    dropout_rate = 0.1
    
    key = jax.random.PRNGKey(10)
    keys = jax.random.split(key, 100)

    params = {
        "w_embedding": init_param(keys[0], (vocab_size, embedding_dim)),
        "w_positional": init_param(keys[1], (1000, embedding_dim)),  # max sequence length of 1000
        "ln_final_gamma": init_param(keys[2], (embedding_dim,)),
        "ln_final_beta": init_param(keys[3], (embedding_dim,)),
        "w_lm_head": init_param(keys[4], (embedding_dim, vocab_size))
    }
    
    for i in range(num_layers):
        base_idx = 5 + i * 10
        params[f'block_{i}'] = {
            'w_q': init_param(keys[base_idx], (embedding_dim, embedding_dim)),
            'w_k': init_param(keys[base_idx+1], (embedding_dim, embedding_dim)),
            'w_v': init_param(keys[base_idx+2], (embedding_dim, embedding_dim)),
            'w_o': init_param(keys[base_idx+3], (embedding_dim, embedding_dim)),
            'w1': init_param(keys[base_idx+4], (embedding_dim, embedding_dim * 4)),
            'b1': jnp.zeros((embedding_dim * 4,)),
            'w2': init_param(keys[base_idx+5], (embedding_dim * 4, embedding_dim)),
            'b2': jnp.zeros((embedding_dim,)),
            'ln1_gamma': init_param(keys[base_idx+6], (embedding_dim,)),
            'ln1_beta': init_param(keys[base_idx+7], (embedding_dim,)),
            'ln2_gamma': init_param(keys[base_idx+8], (embedding_dim,)),
            'ln2_beta': init_param(keys[base_idx+9], (embedding_dim,))
        }
    
    input_seq = jax.random.randint(jax.random.PRNGKey(0), (1, 50), 0, vocab_size)
    output = forward(input_seq, params, num_layers, num_heads, dropout_rate)
    print(f"Output shape: {output.shape}")  # Expected: (1, 50, vocab_size)
