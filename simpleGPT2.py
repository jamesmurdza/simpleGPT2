import numpy as np
from typing import Dict

# Download the hyperparameters and parameters for the pretrained model.
model_size: str = "124M"
models_dir: str = "models"
encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

"""
The hyperparameters look like this:

hparams = {
  "n_vocab": 50257,    # number of tokens in the tokenizer
  "n_ctx": 1024,       # context length of the model
  "n_embd": 768,       # embedding dimension (determines the "width" of the network)
  "n_head": 12,        # number of attention heads (n_embd must be divisible by n_head)
  "n_layer": 12        # number of layers (determines the "depth" of the network)
}
"""

"""
The actual model parameters (weights, biases and encodings) take the following structure and shape.

params = {

    # Matrices for positional and token embeddings:
    "wpe": np.ndarray(shape=(1024, 768), dtype=float32),  # Positional encoding matrix
    "wte": np.ndarray(shape=(50257, 768), dtype=float32),  # Token embedding matrix
    
    # The encoder consists of a bunch of blocks with the same internal structure.
    "blocks": [
        {
            # Inside each block, there's layer normalization.
            "ln_1": {
                "b": np.ndarray(shape=(768,), dtype=float32),  # Bias for layer normalization
                "g": np.ndarray(shape=(768,), dtype=float32)  # Gain for layer normalization
            },
            # Then comes attention and projection.
            "attn": {
                # For attention, we have bias and weight.
                "c_attn": {
                    "b": np.ndarray(shape=(2304,), dtype=float32),  # Bias for attention
                    "w": np.ndarray(shape=(768, 2304), dtype=float32)  # Weight for attention
                },
                # After attention, there's projection.
                "c_proj": {
                    "b": np.ndarray(shape=(768,), dtype=float32),  # Bias for projection
                    "w": np.ndarray(shape=(768, 768), dtype=float32)  # Weight for projection
                }
            },
            # Then, there's also a multi-layer perceptron (feed-forward network).
            "mlp": {
                # It consists of fully connected layers.
                "c_fc": {
                    "b": np.ndarray(shape=(3072,), dtype=float32),  # Bias for the first FC layer
                    "w": np.ndarray(shape=(768, 3072), dtype=float32)  # Weight for the first FC layer
                },
                "c_proj": {
                    "b": np.ndarray(shape=(768,), dtype=float32),  # Bias for the second FC layer
                    "w": np.ndarray(shape=(3072, 768), dtype=float32)  # Weight for the second FC layer
                }
            },
            # Another layer normalization after the feed-forward part.
            "ln_2": {
                "b": np.ndarray(shape=(768,), dtype=float32),  # Bias for layer normalization
                "g": np.ndarray(shape=(768,), dtype=float32)  # Gain for layer normalization
            },
        }
    ],
    # There's layer normalization at the end too.
    "ln_f": {
        "b": np.ndarray(shape=(768,), dtype=float32),  # Bias for layer normalization
        "g": np.ndarray(shape=(768,), dtype=float32)  # Gain for layer normalization
    }
    
}
"""

def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) Activation Function

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Output tensor after applying the GELU activation function.
    """
    # Apply the GELU activation function
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax Activation Function

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Output tensor after applying the softmax activation function.
    """
    # Calculate the exponential of each element and normalize
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization

    Args:
        x (np.ndarray): Input tensor to be normalized.
        g (np.ndarray): Scale parameter (gamma), usually a learned parameter.
        b (np.ndarray): Shift parameter (beta), usually a learned parameter.
        eps (float): A small constant added for numerical stability.

    Returns:
        np.ndarray: The layer-normalized output.
    """
    # Calculate the mean and variance along the last axis
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    # Normalize the input (x) to have zero mean and unit variance
    x_normalized = (x - mean) / np.sqrt(variance + eps)

    # Apply scale (gamma) and shift (beta) parameters
    return g * x_normalized + b


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Linear Transformation

    Args:
        x (np.ndarray): Input tensor of shape [m, in_features].
        w (np.ndarray): Weight matrix of shape [in_features, out_features].
        b (np.ndarray): Bias vector of shape [out_features].

    Returns:
        np.ndarray: Output tensor of shape [m, out_features], resulting from the linear transformation.
    """
    # Perform the linear transformation: multiply input with weight matrix and add bias
    return x @ w + b


def ffn(x: np.ndarray, c_fc: Dict[str, np.ndarray], c_proj: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Feed Forward Network

    Args:
        x (np.ndarray): Input tensor of shape [n_seq, n_embd].
        c_fc (Dict[str, np.ndarray]): Dictionary of weights and biases for the linear projection.
        c_proj (Dict[str, np.ndarray]): Dictionary of weights and biases for the projection back down.

    Returns:
        np.ndarray: Output tensor of shape [n_seq, n_embd].
    """
    # Project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # Project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x

def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Multi-Head Attention

    Args:
        q (np.ndarray): Query tensor of shape [n_q, d_k].
        k (np.ndarray): Key tensor of shape [n_k, d_k].
        v (np.ndarray): Value tensor of shape [n_k, d_v].
        mask (np.ndarray): Attention mask of shape [n_q, n_k].

    Returns:
        np.ndarray: Output tensor of shape [n_q, d_v].
    """
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def main(prompt: str, n_tokens_to_generate: int = 10):
    """
    Generate text tokens based on a given prompt using a Transformer model.

    Args:
        prompt (str): The input text prompt to start generating from.
        n_tokens_to_generate (int): The number of tokens to generate.

    Returns:
        str: The generated text based on the input prompt.
    """
    # Encode the input prompt into a sequence of integers.
    # Dimension: [len(inputs)]
    inputs = encoder.encode(prompt)

    # Ensure the total length of input and tokens to be generated is within the model's max context.
    assert len(inputs) + n_tokens_to_generate < hparams["n_ctx"]

    # Predict each token and append to inputs.
    for _ in range(n_tokens_to_generate):
        # Prepare input embeddings (word + position).
        # Dimensions: [len(inputs), n_embd]
        x = params['wte'][inputs] + params['wpe'][range(len(inputs))]

        # Create a causal mask for attention.
        # Dimension: [len(inputs), len(inputs)]
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

        # Iterate over each transformer block (layer).
        for block in params['blocks']:
            # Apply the first layer normalization.
            # Dimension: [len(inputs), n_embd]
            ln1 = layer_norm(x, **block['ln_1'])

            # Compute query, key, value for attention using a linear transformation.
            # Dimension: [len(inputs), 3 * n_embd]
            qkv = linear(ln1, **block['attn']['c_attn'])

            # Split the qkv into separate heads and process each head.
            # Each head dimension: [len(inputs), n_embd / n_head]
            qkv_heads = np.split(qkv, 3*hparams['n_head'], axis=-1)

            attn_out = []
            for head_id in range(hparams['n_head']):
                # Perform scaled dot-product attention for each head.
                # Output dimension per head: [len(inputs), n_embd / n_head]
                out = attention(qkv_heads[head_id],
                                qkv_heads[head_id + hparams['n_head']],
                                qkv_heads[head_id + 2*hparams['n_head']],
                                causal_mask)
                attn_out.append(out)

            # Merge output from all heads.
            # Dimension: [len(inputs), n_embd]
            attn = linear(np.hstack(attn_out), **block['attn']['c_proj'])

            # Add the attention output to the original input (residual connection).
            # Dimension: [len(inputs), n_embd]
            x = x + attn

            # Apply the second layer normalization.
            # Dimension: [len(inputs), n_embd]
            ln2 = layer_norm(x, **block['ln_2'])

            # Apply the feed-forward network (MLP).
            # Dimension: [len(inputs), n_embd]
            ffn_out = ffn(ln2, **block['mlp'])

            # Add the feed-forward output to the original input (residual connection).
            # Dimension: [len(inputs), n_embd]
            x = x + ffn_out

        # Apply the final layer normalization and project back to vocabulary space.
        # Dimension of logits: [n_vocab]
        logits = layer_norm(x[-1], **params['ln_f']) @ params['wte'].T

        # Choose the token with the highest probability.
        next_id = np.argmax(logits)
        print(encoder.decode([int(next_id)]), end="", flush=True)

        # Append the generated token to the input sequence.
        inputs.append(int(next_id))

    # Decode the generated tokens back into text.
    output_ids = inputs[len(inputs) - n_tokens_to_generate:]
    output_text = encoder.decode(output_ids)

if __name__ == "__main__":
    main("Alan Turing theorized that computers would one day become", n_tokens_to_generate=10)
