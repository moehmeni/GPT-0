import jax.numpy as jnp
import jax

# ---------- Utils ----------


def init_param(key, shape: tuple):
    # He initialization
    n_prev = shape[1]
    std = jnp.sqrt(2 / n_prev)
    gaussian_scaled = jax.random.normal(key, shape) * std
    return gaussian_scaled


def softmax(x):
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)


def dropout(key, z: jnp.ndarray, drop_rate: float, train: bool = True):
    if not train:
        return z
    mask = jax.random.binomial(key, 1, 1 - drop_rate, z.shape) / drop_rate
    return z * mask


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


# ---------- Transformers Utils ----------


def embedding(x, w_e):
    return x @ w_e


def positional_encoding(x, w_p):
    return x + w_p


def self_attention(q, k, v):
    scale = 1 / jnp.sqrt(q.shape[-1])
    return softmax(q @ k.T * scale) @ v


def forward(x, params: dict):
    pass


if __name__ == "__main__":
    vocab_size = 1000
    embedding_dim = 512

    key = jax.random.PRNGKey(10)
    params = {"w_embedding": init_param((vocab_size, embedding_dim)), "w_": 2}
