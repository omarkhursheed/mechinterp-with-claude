import numpy as np

# Core functions
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def softmax_backward(grad_output, softmax_output):
    s = softmax_output
    grad_input = s * grad_output
    grad_input -= s * np.sum(grad_output * s)
    return grad_input

class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        
    def forward(self, x):
        self.x = x
        n = len(x)
        self.scores = x @ x.T
        self.attn_weights = np.zeros_like(self.scores)
        for i in range(n):
            self.attn_weights[i] = softmax(self.scores[i])
        self.output = self.attn_weights @ x
        return self.output
    
    def backward(self, grad_output):
        n = len(self.x)
        grad_attn_weights = grad_output @ self.x.T  
        grad_x_from_output = self.attn_weights.T @ grad_output
        grad_scores = np.zeros_like(self.scores)
        for i in range(n):
            grad_scores[i] = softmax_backward(grad_attn_weights[i], self.attn_weights[i])
        grad_x_from_first = grad_scores @ self.x
        grad_x_from_second = grad_scores.T @ self.x
        grad_x = grad_x_from_output + grad_x_from_first + grad_x_from_second
        return grad_x

# Your FeedForward with backward from yesterday
class FeedForward:
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.W1 = np.random.randn(d_model, self.d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * np.sqrt(2.0 / self.d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        self.x = x  
        self.z1 = self.x @ self.W1 + self.b1
        self.hidden = np.maximum(0, self.z1)  
        return self.hidden @ self.W2 + self.b2
    
    def backward(self, grad_output):
        self.grad_W2 = self.hidden.T @ grad_output
        self.grad_b2 = grad_output.sum(axis=0)
        grad_hidden = grad_output @ self.W2.T
        grad_z1 = grad_hidden * (self.z1 > 0)
        self.grad_W1 = self.x.T @ grad_z1
        self.grad_b1 = grad_z1.sum(axis=0)
        return grad_z1 @ self.W1.T

