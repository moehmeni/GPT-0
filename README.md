# GPT-0  
This project is a minimal implementation of a Generative Pre-trained Transformer (GPT), using NumPy for core operations and JAX to handle backpropagation. The goal is to build a simplified version of the GPT model from scratch, focusing on understanding how the architecture works at its core.

The model implements key features of the GPT architecture, such as the attention mechanism and transformer blocks, allowing it to process and generate sequences of text. By using NumPy for the main computations, the project emphasizes a hands-on approach to building the model, while JAX is used to simplify gradient calculation and optimization during training.

### Visual Guide  
Below is a diagram of the full GPT architecture, which this project is based on:

<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Full_GPT_architecture.png" width="350px"/>

### TODO
- [ ] Causal masking for autoregressive prediction  
- [ ] Tokenization and text generation utilities  
- [ ] Training loop and optimization strategies
