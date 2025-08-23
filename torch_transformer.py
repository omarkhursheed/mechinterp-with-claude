import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = torch.nn.Linear(self.d_model, self.d_model)
        self.k = torch.nn.Linear(self.d_model, self.d_model)
        self.v = torch.nn.Linear(self.d_model, self.d_model)
        self.scale = self.d_head ** 0.5
        self.output = torch.nn.Linear(self.d_model, self.d_model)
        
    def forward(self, res):
        batch_size, seq_len, d_model = res.shape
        q = self.q(res)  # 
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        q = q.transpose(1,2)
        k = self.k(res)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.transpose(1,2)
        v = self.v(res)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.transpose(1,2) # b, ,n_head, seq, d_head @ b, n_head, d_head, seq -> b, n_head, seq, seq 
        logits = q @ k.transpose(-2,-1) / self.scale
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().to(res.device)
        logits = logits.masked_fill(mask,-float("Inf"))
        attn = F.softmax(logits, dim=-1)
        y = attn @ v # b, n_head, seq, d_head
        y = y.transpose(1,2)
        return self.output(y.reshape(batch_size, seq_len, self.d_model)), attn
    
# Test the module
mha = MultiHeadAttention(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
output, attention = mha(x)
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        # What parameters do we need here?
        
    def forward(self, x):
        # Compute mean and std
        # Normalize
        # Apply learnable parameters
        return (x - x.mean(dim=-1,keepdim=True))/ (x.var(dim=-1, keepdim=True)+self.eps).sqrt() * self.weight + self.bias

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_model * 4
        self.ln1 = LayerNorm(self.d_model)
        self.mha = MultiHeadAttention(self.d_model, self.n_heads)
        self.ln2 = LayerNorm(self.d_model)
        self.ffn1 = nn.Linear(self.d_model, self.d_ffn)
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(self.d_ffn, self.d_model)
        
        
    def forward(self, x):
        ln1 = self.ln1(x)
        scores, attn = self.mha(ln1)
        curr = x + scores

        ln2 = self.ln2(curr)
        ffn_op =self.ffn2(self.act(self.ffn1(ln2)))

        op = curr + ffn_op
        return op, attn
    # Step 1: ?

    # Step 2: ?

    # etc.

   
# Test your TransformerBlock
block = TransformerBlock(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)
output, attn = block(x)
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attn.shape}")

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.ones(d_model))
        module_list = []
        for _ in range(n_layers):
            module_list.append(TransformerBlock(self.d_model, self.n_heads))
        self.modules = nn.ModuleList(module_list)
        self.W_U = nn.Embedding(d_model, vocab_size)

        
        # Stack of transformer blocks - how?
        
        # Final output - what do we need?
        
    def forward(self, input_ids):
        # Convert tokens to embeddings
        # Add positional information
        # Pass through transformer blocks
        # Convert to logits
        pass