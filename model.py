import jax.numpy as jnp
import jax


def init_param(shape: tuple):
    # He initialization
    n_prev = shape[1]
    std = jnp.sqrt(2 / n_prev)
    gaussian_scaled = jax.random.normal(key, shape) * std
    return gaussian_scaled


def forward(x, params : dict):
    
    
if __name__ == "__main__":
    vocab_size = 1000
    embedding_dim = 512

    key = jax.random.PRNGKey(10)
    params = {
        "w_embedding": init_param((embedding_dim, vocab_size)),
        ""
    }
    print("Hi")
