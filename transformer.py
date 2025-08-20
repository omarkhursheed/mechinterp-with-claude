"""
Transformer from Scratch in NumPy
Built from first principles - no deep learning libraries required!
Author: [Your name]
Date: [Today's date]

This implementation includes:
- Multi-head self-attention with causal masking
- Layer normalization
- Feedforward networks
- Positional encodings
- A complete mini-GPT architecture
"""

import numpy as np

# =============================================================================
# Core Components
# =============================================================================

def softmax(x):
    """Compute softmax values for array x along last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):
    """
    Apply layer normalization.
    Normalizes inputs to have mean=0 and variance=1 across features.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

# =============================================================================
# Attention Mechanisms
# =============================================================================

def single_query_attention(query, keys, values):
    """
    Basic attention for a single query vector.
    
    Args:
        query: [d] vector
        keys: [n, d] matrix of keys
        values: [n, d] matrix of values
    
    Returns:
        output: [d] weighted combination of values
        weights: [n] attention weights
    """
    # Compute attention scores via dot product
    scores = query @ keys.T  # [n]
    
    # Convert to probabilities
    weights = softmax(scores)
    
    # Weighted average of values
    output = weights @ values  # [d]
    
    return output, weights

def self_attention(embeddings, mask=None):
    """
    Self-attention where every position attends to every other position.
    
    Args:
        embeddings: [n, d] matrix where n is sequence length, d is dimension
        mask: Optional [n, n] mask matrix (0s where attention is blocked)
    
    Returns:
        outputs: [n, d] new representations
        attention_weights: [n, n] attention matrix
    """
    n, d = embeddings.shape
    
    # All positions serve as queries, keys, and values
    scores = embeddings @ embeddings.T  # [n, n]
    
    # Apply mask if provided (for causal attention)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Row-wise softmax (each position's attention sums to 1)
    attention_weights = np.zeros((n, n))
    for i in range(n):
        attention_weights[i] = softmax(scores[i])
    
    # Apply attention to get new representations
    outputs = attention_weights @ embeddings  # [n, d]
    
    return outputs, attention_weights

# =============================================================================
# Transformer Components
# =============================================================================

class FeedForward:
    """
    Two-layer feedforward network with ReLU activation.
    Expands dimension by 4x then projects back.
    """
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, self.d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * np.sqrt(2.0 / self.d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """Forward pass through feedforward network."""
        # First linear + ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        # Second linear
        output = hidden @ self.W2 + self.b2
        return output

class TransformerBlock:
    """
    A single transformer block consisting of:
    1. Self-attention with residual connection and layer norm
    2. Feedforward with residual connection and layer norm
    """
    def __init__(self, d_model):
        self.d_model = d_model
        self.feedforward = FeedForward(d_model)
    
    def forward(self, x, mask=None):
        """
        Process input through transformer block.
        
        Args:
            x: [n, d_model] input embeddings
            mask: Optional causal mask
        
        Returns:
            [n, d_model] transformed embeddings
        """
        # Self-attention with residual
        attn_out, attn_weights = self_attention(x, mask)
        x = x + attn_out
        x = layer_norm(x)
        
        # Feedforward with residual
        ff_out = self.feedforward.forward(x)
        x = x + ff_out
        x = layer_norm(x)
        
        return x, attn_weights

# =============================================================================
# Mini GPT Model
# =============================================================================

class MiniGPT:
    """
    A minimal GPT implementation with:
    - Token embeddings
    - Positional encodings
    - Single transformer block
    - Output projection to vocabulary
    """
    def __init__(self, vocab_size=10, d_model=64, n_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embeddings
        self.embed_matrix = np.random.randn(vocab_size, d_model) * 0.1
        
        # Positional encodings (learned)
        self.max_seq_len = 512
        self.position_embeds = np.random.randn(self.max_seq_len, d_model) * 0.1
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model) for _ in range(n_layers)]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.1
    
    def forward(self, tokens, return_attention=False):
        """
        Forward pass through the model.
        
        Args:
            tokens: [n] array of token indices
            return_attention: Whether to return attention weights
        
        Returns:
            logits: [n, vocab_size] predictions
            attention_weights: Optional list of attention matrices from each layer
        """
        seq_len = len(tokens)
        
        # Token embeddings
        x = self.embed_matrix[tokens]  # [n, d_model]
        
        # Add positional encodings
        x = x + self.position_embeds[:seq_len]
        
        # Create causal mask (lower triangular)
        mask = np.tril(np.ones((seq_len, seq_len)))
        
        # Process through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_w = block.forward(x, mask)
            attention_weights.append(attn_w)
        
        # Project to vocabulary
        logits = x @ self.output_proj  # [n, vocab_size]
        
        if return_attention:
            return logits, attention_weights
        return logits

# =============================================================================
# Training Utilities
# =============================================================================

def compute_loss(logits, targets):
    """
    Compute cross-entropy loss.
    
    Args:
        logits: [n, vocab_size] predictions
        targets: [n] correct token indices
    
    Returns:
        scalar loss value
    """
    n = logits.shape[0]
    
    # Convert logits to probabilities
    probs = softmax(logits)
    
    # Get probability of correct tokens
    correct_probs = probs[np.arange(n), targets]
    
    # Cross-entropy loss
    loss = -np.mean(np.log(correct_probs + 1e-10))  # Add epsilon for stability
    
    return loss

def generate_sequence_reversal_data(seq_len=4, vocab_size=10):
    """Generate a random sequence and its reversal for training."""
    input_seq = np.random.randint(0, vocab_size, seq_len)
    target_seq = input_seq[::-1]  # Reverse the sequence
    return input_seq, target_seq

# =============================================================================
# Demo and Testing
# =============================================================================

def demo():
    """Demonstrate the transformer on sequence reversal task."""
    print("="*60)
    print("TRANSFORMER FROM SCRATCH - DEMO")
    print("="*60)
    
    # Initialize model
    model = MiniGPT(vocab_size=10, d_model=32, n_layers=2)
    
    # Generate sample data
    input_tokens = np.array([1, 2, 3, 4])
    target_tokens = np.array([4, 3, 2, 1])
    
    print(f"\nInput sequence:  {input_tokens}")
    print(f"Target sequence: {target_tokens}")
    
    # Forward pass
    logits, attention_weights = model.forward(input_tokens, return_attention=True)
    
    # Compute loss
    loss = compute_loss(logits, target_tokens)
    
    # Get predictions
    predictions = np.argmax(logits, axis=-1)
    
    print(f"\nModel predictions: {predictions}")
    print(f"Cross-entropy loss: {loss:.4f}")
    print(f"Random baseline loss: {-np.log(1/10):.4f}")
    
    # Show attention pattern from last layer
    print("\nAttention pattern (last layer):")
    attn = attention_weights[-1]
    for i in range(len(input_tokens)):
        weights = attn[i]
        attended_positions = [j for j, w in enumerate(weights) if w > 0.01]
        print(f"  Position {i} attends to positions: {attended_positions}")
    
    return model, loss

def visualize_attention(attention_weights, tokens=None):
    """Simple text visualization of attention patterns."""
    n = attention_weights.shape[0]
    
    if tokens is None:
        tokens = [str(i) for i in range(n)]
    
    print("\nAttention Matrix:")
    print("     ", "  ".join(f"{t:^5}" for t in tokens))
    print("     " + "-"*6*n)
    
    for i, token in enumerate(tokens):
        row = attention_weights[i]
        # Convert to percentages
        row_str = "  ".join(f"{val:5.1%}" for val in row)
        print(f"{token:^5}| {row_str}")

# =============================================================================
# Main Execution
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

# First, let's modify your MiniGPT class slightly to capture intermediate activations
class MiniGPTWithHooks(MiniGPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = {}  # Store intermediate activations
    
    def forward(self, tokens, return_attention=False, capture_activations=False):
        """Modified forward to capture intermediate states"""
        seq_len = len(tokens)
        
        # Token embeddings
        x = self.embed_matrix[tokens]
        if capture_activations:
            self.activations['embeddings'] = x.copy()
        
        # Add positional encodings
        x = x + self.position_embeds[:seq_len]
        
        # Create causal mask
        mask = np.tril(np.ones((seq_len, seq_len)))
        
        # Process through transformer blocks
        attention_weights = []
        for i, block in enumerate(self.blocks):
            x, attn_w = block.forward(x, mask)
            attention_weights.append(attn_w)
            if capture_activations:
                self.activations[f'block_{i}_output'] = x.copy()
                self.activations[f'block_{i}_attention'] = attn_w.copy()
        
        # Project to vocabulary
        logits = x @ self.output_proj
        
        if return_attention:
            return logits, attention_weights
        return logits

# Create your model with hooks
model = MiniGPTWithHooks(vocab_size=10, d_model=32, n_layers=2)

# Now let's visualize attention patterns for different inputs
def explore_attention_patterns(model):
    """See what patterns your random model learned"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    test_inputs = [
        ([1, 2, 3, 4], "Sequential"),
        ([1, 1, 1, 1], "Repeated"),
        ([5, 6, 5, 6], "Alternating"),
        ([9, 0, 0, 9], "Bookended"),
    ]
    
    for idx, (tokens, label) in enumerate(test_inputs):
        tokens = np.array(tokens)
        _, attn_weights = model.forward(tokens, return_attention=True)
        
        # Plot layer 0 attention
        ax = axes[0, idx]
        im = ax.imshow(attn_weights[0], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"Layer 0: {label}")
        ax.set_ylabel("To position")
        
        # Plot layer 1 attention
        ax = axes[1, idx]
        im = ax.imshow(attn_weights[1], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"Layer 1: {label}")
        ax.set_xlabel("From position")
        ax.set_ylabel("To position")
        
        # Add colorbar for last plot
        if idx == 3:
            plt.colorbar(im, ax=ax)
    
    plt.suptitle("Attention Patterns Across Layers")
    plt.tight_layout()
    plt.show()
def find_neuron_activators(model, layer=0, neuron_idx=0, n_samples=1000):
    """Find what inputs maximally activate a specific neuron"""
    
    max_activations = []
    best_inputs = []
    
    for _ in range(n_samples):
        # Random input
        tokens = np.random.randint(0, 10, 4)
        
        # Forward pass with activation capture
        _ = model.forward(tokens, capture_activations=True)
        
        # Get activation of specific neuron
        if layer == 0:
            activation = model.activations['block_0_output'][:, neuron_idx].max()
        else:
            activation = model.activations['block_1_output'][:, neuron_idx].max()
        
        max_activations.append(activation)
        best_inputs.append(tokens)
    
    # Get top 5 activating inputs
    top_indices = np.argsort(max_activations)[-5:]
    
    print(f"Top inputs for Layer {layer}, Neuron {neuron_idx}:")
    for idx in top_indices:
        print(f"  Input {best_inputs[idx]} → activation {max_activations[idx]:.3f}")
    
    return best_inputs, max_activations
# Let's find neurons with interesting behaviors
def analyze_neuron_selectivity(model, n_samples=500):
    """Find neurons that are selective for specific patterns"""
    
    # We'll track what makes each neuron fire
    neuron_stats = {
        'position_selective': [],  # Fires for specific positions
        'token_selective': [],     # Fires for specific tokens
        'pattern_selective': []    # Fires for specific patterns
    }
    
    # Test neurons 0-10 in layer 0
    for neuron_idx in range(10):
        position_activations = {0: [], 1: [], 2: [], 3: []}
        token_activations = {i: [] for i in range(10)}
        
        for _ in range(n_samples):
            tokens = np.random.randint(0, 10, 4)
            _ = model.forward(tokens, capture_activations=True)
            
            activations = model.activations['block_0_output']
            
            # Track activation by position
            for pos in range(4):
                position_activations[pos].append(activations[pos, neuron_idx])
            
            # Track activation by token value
            for pos, token in enumerate(tokens):
                token_activations[token].append(activations[pos, neuron_idx])
        
        # Check if neuron is position-selective
        mean_by_position = [np.mean(position_activations[p]) for p in range(4)]
        if np.std(mean_by_position) > 0.5:  # High variance across positions
            neuron_stats['position_selective'].append((neuron_idx, mean_by_position))
        
        # Check if neuron is token-selective  
        mean_by_token = [np.mean(token_activations[t]) if token_activations[t] else 0 
                         for t in range(10)]
        if np.std(mean_by_token) > 0.5:  # High variance across tokens
            neuron_stats['token_selective'].append((neuron_idx, mean_by_token))
    
    return neuron_stats

# Run the analysis
stats = analyze_neuron_selectivity(model)

# Visualize selective neurons
if stats['position_selective']:
    print("Position-selective neurons found:")
    for neuron_idx, means in stats['position_selective'][:3]:  # Show top 3
        print(f"  Neuron {neuron_idx}: Position preferences = {[f'{m:.2f}' for m in means]}")

if stats['token_selective']:
    print("\nToken-selective neurons found:")
    for neuron_idx, means in stats['token_selective'][:3]:
        top_tokens = np.argsort(means)[-3:]
        print(f"  Neuron {neuron_idx}: Prefers tokens {top_tokens}")
# Try this for different neurons
# inputs, acts = find_neuron_activators(model, layer=0, neuron_idx=0)
# Run this!
# explore_attention_patterns(model)
# if __name__ == "__main__":
#     # Run demo
#     model, initial_loss = demo()
    
#     # Visualize attention for a simple example
#     print("\n" + "="*60)
#     print("ATTENTION VISUALIZATION")
#     print("="*60)
    
#     tokens = np.array([5, 2, 8, 1])
#     logits, attention_weights = model.forward(tokens, return_attention=True)
    
#     # Show attention from first layer
#     visualize_attention(attention_weights[0], 
#                         tokens=[f"tok_{t}" for t in tokens])
    
#     print("\n" + "="*60)
#     print("IMPLEMENTATION COMPLETE!")
#     print("="*60)
#     print("\nYou've successfully built a transformer from scratch!")
#     print("Next steps:")
#     print("  1. Add backpropagation to make it trainable")
#     print("  2. Implement multi-head attention")
#     print("  3. Add dropout and other regularization")
#     print("  4. Scale up to larger models")
#     print("  5. Try different tasks beyond sequence reversal")