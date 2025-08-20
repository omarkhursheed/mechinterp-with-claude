# Try it:
x = 3
b = 1

w = 2.1
y = x * w + b
loss_bigger = y ** 2
print(f"w=2.1: y={y}, loss={loss_bigger}")

w = 1.9
y = x * w + b  
loss_smaller = y ** 2
print(f"w=1.9: y={y}, loss={loss_smaller}")
class TinyLinear:
    def init(self):
        self.w = 2.0
        self.b = 1.0

    def forward(self, x):
        self.x = x  # Save for backward!
        self.y = x * self.w + self.b
        return self.y

    def backward(self, grad_y):
        """
        grad_y: how much the loss changes when y changes (dloss/dy)
        """
        # Question: if y = x*w + b, what are:
        self.grad_w = grad_y * x  # How much does y change when w changes?
        self.grad_b = grad_y * 1  # How much does y change when b changes?

        # And what gradient should we pass back to whoever gave us x?
        grad_x = grad_y * self.w
        return grad_x
# What if we have: x -> Linear1 -> y -> Linear2 -> z -> loss
linear1 = TinyLinear()  # x -> y
linear2 = TinyLinear()  # y -> z

x = 3
y = linear1.forward(x)  
z = linear2.forward(y)
loss = z ** 2

# Now backprop:
grad_z = 2 * z                    # gradient of loss w.r.t. z
grad_y = linear2.backward(grad_z)  # linear2 tells us: "if y changes, here's how z changes"
grad_x = linear1.backward(grad_y)  # linear1 tells us: "if x changes, here's how y changes"
class TinyLinear:
    def __init__(self):  # <- double underscores!
        self.w = 2.0
        self.b = 1.0
    
    def forward(self, x):
        self.x = x  # Save for backward!
        self.y = x * self.w + self.b
        return self.y
    
    def backward(self, grad_y):
        """
        grad_y: how much the loss changes when y changes (dloss/dy)
        """
        self.grad_w = grad_y * self.x  # <- also fixed: use self.x
        self.grad_b = grad_y * 1
        
        grad_x = grad_y * self.w
        return grad_x
# What if we have: x -> Linear1 -> y -> Linear2 -> z -> loss
linear1 = TinyLinear()  # x -> y
linear2 = TinyLinear()  # y -> z

x = 3
y = linear1.forward(x)  
z = linear2.forward(y)
loss = z ** 2

# Now backprop:
grad_z = 2 * z                    # gradient of loss w.r.t. z
grad_y = linear2.backward(grad_z)  # linear2 tells us: "if y changes, here's how z changes"
grad_x = linear1.backward(grad_y)  # linear1 tells us: "if x changes, here's how y changes"
# Test it
linear = TinyLinear()
x = 3
y = linear.forward(x)
print(f"Forward: x={x} -> y={y}")

# Pretend the loss wants y to decrease (negative gradient)
grad_y = -1  
grad_x = linear.backward(grad_y)

print(f"Backward: grad_y={grad_y}")
print(f"  -> grad_w={linear.grad_w}")  
print(f"  -> grad_b={linear.grad_b}")
print(f"  -> grad_x={grad_x}")
class TinyReLU:

    def forward(self, x):

        self.x = x  # Save for backward

        self.y = np.maximum(0, x)

        return self.y

    

    def backward(self, grad_y):

        # ReLU backward: gradient flows where forward > 0, blocked where forward <= 0

        # Question: How do you write this as a mask?

        grad_x = grad_y * (self.x > 0)

        return grad_x
relu = TinyReLU()

# Test positive input
x = 3
y = relu.forward(x)
grad_x = relu.backward(1)  # gradient of 1 coming from above
print(f"x={x} -> y={y}, grad flows back: {grad_x}")

# Test negative input  
x = -2
y = relu.forward(x)
grad_x = relu.backward(1)
print(f"x={x} -> y={y}, grad flows back: {grad_x}")
import numpy as np
relu = TinyReLU()

# Test positive input
x = 3
y = relu.forward(x)
grad_x = relu.backward(1)  # gradient of 1 coming from above
print(f"x={x} -> y={y}, grad flows back: {grad_x}")

# Test negative input  
x = -2
y = relu.forward(x)
grad_x = relu.backward(1)
print(f"x={x} -> y={y}, grad flows back: {grad_x}")
A = np.random.randn(2, 3)
B = np.random.randn(3, 4)
C = A @ B
grad_C = np.ones_like(C)  # (2, 4)

grad_B = A.T @ grad_C  # (3, 2) @ (2, 4) = (3, 4) ✓
print(f"B shape: {B.shape}, grad_B shape: {grad_B.shape}")  # Should match!
class TinyTwoLayerNet:

    def init(self, input_dim=3, hidden_dim=4, output_dim=2):

        # First layer: input_dim -> hidden_dim

        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.5

        self.b1 = np.zeros(hidden_dim)

        

        # Second layer: hidden_dim -> output_dim

        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.5

        self.b2 = np.zeros(output_dim)

        

        # Initialize gradients

        self.grad_w1 = None

        self.grad_b1 = None

        self.grad_w2 = None

        self.grad_b2 = None

    

    def forward(self, x):

        # Layer 1: Linear

        self.x = x

        self.z1 = x @ self.w1 + self.b1

        

        # ReLU

        self.h = np.maximum(0, self.z1)

        

        # Layer 2: Linear

        self.y = self.h @ self.w2 + self.b2

        

        return self.y

    

def backward(self, grad_y):

    """

    grad_y: gradient of loss w.r.t output y

    """

    # Layer 2 gradients (y = h @ w2 + b2)

    self.grad_w2 = self.h.T @ grad_y  

    self.grad_b2 = grad_y.sum(axis=0) 

    

    # Gradient flowing back to h

    grad_h = grad_y @ self.w2.T  # Fixed dimension order

    

    # ReLU gradient

    grad_z1 = grad_h * (self.z1 > 0)  # Simplified

    

    # Layer 1 gradients

    self.grad_w1 = self.x.T @ grad_z1  # Perfect!

    self.grad_b1 = grad_z1.sum(axis=0)  # Sum over batch

    

    # Gradient w.r.t input

    grad_x = grad_z1 @ self.w1.T

    return grad_x
class TinyTwoLayerNet:

    def init(self, input_dim=3, hidden_dim=4, output_dim=2):

        # First layer: input_dim -> hidden_dim

        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.5

        self.b1 = np.zeros(hidden_dim)

        # Second layer: hidden_dim -> output_dim

        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.5

        self.b2 = np.zeros(output_dim)

    def forward(self, x):

        # Layer 1: Linear

        self.x = x

        self.z1 = x @ self.w1 + self.b1

        # ReLU

        self.h = np.maximum(0, self.z1)

        # Layer 2: Linear

        self.y = self.h @ self.w2 + self.b2

        return self.y

    def backward(self, grad_y):

        # Work backwards through what forward did

        # Layer 2 gradients (y = h @ w2 + b2)

        self.grad_w2 = self.h.T @ grad_y  # ✓ This one is correct!

        self.grad_b2 = grad_y.sum(axis=0) # ✓ This one is correct!

        # What gradient flows back to h?

        # Forward was: y = h @ w2 + b2

        # You just learned: if C = A @ B, then grad_A = grad_C @ B.T

        grad_h =  grad_y @ self.w2.T         

        # ReLU gradient

        # Remember: ReLU lets gradient through where forward was > 0

        # Which variable should we check > 0?

        grad_z1 = grad_h * (self.z1 > 0)        

        # Layer 1 gradients (z1 = x @ w1 + b1)

        # Forward was: z1 = x @ w1 + b1

        # Apply the pattern you just learned!

        grad_w1 = self.x.T @ grad_z1

        grad_b1 =  grad_z1.sum(axis=0)# FIX THIS (hint: same pattern as grad_b2)

        # Don't need grad_x for this simple example

        return grad_z1 @ self.w1.T
# Create a simple learning problem
np.random.seed(42)
net = TinyTwoLayerNet(input_dim=2, hidden_dim=3, output_dim=1)

# XOR-like problem
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])
y_true = np.array([[0], [1], [1], [0]])  # XOR

# Training loop!
learning_rate = 0.1
losses = []

for step in range(100):
    # Forward
    y_pred = net.forward(X)
    
    # Compute loss (MSE)
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)
    
    # Backward
    grad_y = 2 * (y_pred - y_true) / len(X)  # MSE gradient
    net.backward(grad_y)
    
    # Update weights! (This is gradient descent)
    net.w1 -= learning_rate * net.grad_w1
    net.b1 -= learning_rate * net.grad_b1
    net.w2 -= learning_rate * net.grad_w2
    net.b2 -= learning_rate * net.grad_b2
    
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

# Test it!
print("\nFinal predictions:")
y_final = net.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {y_final[i,0]:.3f}, True: {y_true[i,0]}")
class TinyTwoLayerNet:

    def __init__(self, input_dim=3, hidden_dim=4, output_dim=2):

        # First layer: input_dim -> hidden_dim

        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.5

        self.b1 = np.zeros(hidden_dim)

        # Second layer: hidden_dim -> output_dim

        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.5

        self.b2 = np.zeros(output_dim)

    def forward(self, x):

        # Layer 1: Linear

        self.x = x

        self.z1 = x @ self.w1 + self.b1

        # ReLU

        self.h = np.maximum(0, self.z1)

        # Layer 2: Linear

        self.y = self.h @ self.w2 + self.b2

        return self.y

    def backward(self, grad_y):

        # Work backwards through what forward did

        # Layer 2 gradients (y = h @ w2 + b2)

        self.grad_w2 = self.h.T @ grad_y  # ✓ This one is correct!

        self.grad_b2 = grad_y.sum(axis=0) # ✓ This one is correct!

        # What gradient flows back to h?

        # Forward was: y = h @ w2 + b2

        # You just learned: if C = A @ B, then grad_A = grad_C @ B.T

        grad_h =  grad_y @ self.w2.T         

        # ReLU gradient

        # Remember: ReLU lets gradient through where forward was > 0

        # Which variable should we check > 0?

        grad_z1 = grad_h * (self.z1 > 0)        

        # Layer 1 gradients (z1 = x @ w1 + b1)

        # Forward was: z1 = x @ w1 + b1

        # Apply the pattern you just learned!

        grad_w1 = self.x.T @ grad_z1

        grad_b1 =  grad_z1.sum(axis=0)# FIX THIS (hint: same pattern as grad_b2)

        # Don't need grad_x for this simple example

        return grad_z1 @ self.w1.T
# Create a simple learning problem
np.random.seed(42)
net = TinyTwoLayerNet(input_dim=2, hidden_dim=3, output_dim=1)

# XOR-like problem
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])
y_true = np.array([[0], [1], [1], [0]])  # XOR

# Training loop!
learning_rate = 0.1
losses = []

for step in range(100):
    # Forward
    y_pred = net.forward(X)
    
    # Compute loss (MSE)
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)
    
    # Backward
    grad_y = 2 * (y_pred - y_true) / len(X)  # MSE gradient
    net.backward(grad_y)
    
    # Update weights! (This is gradient descent)
    net.w1 -= learning_rate * net.grad_w1
    net.b1 -= learning_rate * net.grad_b1
    net.w2 -= learning_rate * net.grad_w2
    net.b2 -= learning_rate * net.grad_b2
    
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

# Test it!
print("\nFinal predictions:")
y_final = net.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {y_final[i,0]:.3f}, True: {y_true[i,0]}")
class TinyTwoLayerNet:

    def __init__(self, input_dim=3, hidden_dim=4, output_dim=2):

        # First layer: input_dim -> hidden_dim

        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.5

        self.b1 = np.zeros(hidden_dim)

        # Second layer: hidden_dim -> output_dim

        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.5

        self.b2 = np.zeros(output_dim)

    def forward(self, x):

        # Layer 1: Linear

        self.x = x

        self.z1 = x @ self.w1 + self.b1

        # ReLU

        self.h = np.maximum(0, self.z1)

        # Layer 2: Linear

        self.y = self.h @ self.w2 + self.b2

        return self.y

    def backward(self, grad_y):

        # Work backwards through what forward did

        # Layer 2 gradients (y = h @ w2 + b2)

        self.grad_w2 = self.h.T @ grad_y  # ✓ This one is correct!

        self.grad_b2 = grad_y.sum(axis=0) # ✓ This one is correct!

        # What gradient flows back to h?

        # Forward was: y = h @ w2 + b2

        # You just learned: if C = A @ B, then grad_A = grad_C @ B.T

        self.grad_h =  grad_y @ self.w2.T         

        # ReLU gradient

        # Remember: ReLU lets gradient through where forward was > 0

        # Which variable should we check > 0?

        self.grad_z1 = self.grad_h * (self.z1 > 0)        

        # Layer 1 gradients (z1 = x @ w1 + b1)

        # Forward was: z1 = x @ w1 + b1

        # Apply the pattern you just learned!

        self.grad_w1 = self.x.T @ self.grad_z1

        self.grad_b1 =  self.grad_z1.sum(axis=0)# FIX THIS (hint: same pattern as grad_b2)

        # Don't need grad_x for this simple example

        return self.grad_z1 @ self.w1.T
# Create a simple learning problem
np.random.seed(42)
net = TinyTwoLayerNet(input_dim=2, hidden_dim=3, output_dim=1)

# XOR-like problem
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])
y_true = np.array([[0], [1], [1], [0]])  # XOR

# Training loop!
learning_rate = 0.1
losses = []

for step in range(100):
    # Forward
    y_pred = net.forward(X)
    
    # Compute loss (MSE)
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)
    
    # Backward
    grad_y = 2 * (y_pred - y_true) / len(X)  # MSE gradient
    net.backward(grad_y)
    
    # Update weights! (This is gradient descent)
    net.w1 -= learning_rate * net.grad_w1
    net.b1 -= learning_rate * net.grad_b1
    net.w2 -= learning_rate * net.grad_w2
    net.b2 -= learning_rate * net.grad_b2
    
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

# Test it!
print("\nFinal predictions:")
y_final = net.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {y_final[i,0]:.3f}, True: {y_true[i,0]}")
# Create a simple learning problem
np.random.seed(42)
net = TinyTwoLayerNet(input_dim=2, hidden_dim=3, output_dim=1)

# XOR-like problem
X = np.array([[0, 0],
              [0, 1], 
              [1, 0],
              [1, 1]])
y_true = np.array([[0], [1], [1], [0]])  # XOR

# Training loop!
learning_rate = 0.1
losses = []

for step in range(1000):
    # Forward
    y_pred = net.forward(X)
    
    # Compute loss (MSE)
    loss = np.mean((y_pred - y_true) ** 2)
    losses.append(loss)
    
    # Backward
    grad_y = 2 * (y_pred - y_true) / len(X)  # MSE gradient
    net.backward(grad_y)
    
    # Update weights! (This is gradient descent)
    net.w1 -= learning_rate * net.grad_w1
    net.b1 -= learning_rate * net.grad_b1
    net.w2 -= learning_rate * net.grad_w2
    net.b2 -= learning_rate * net.grad_b2
    
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

# Test it!
print("\nFinal predictions:")
y_final = net.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {y_final[i,0]:.3f}, True: {y_true[i,0]}")
def softmax_backward(grad_output, softmax_output):
    """
    grad_output: gradient of loss w.r.t. softmax output
    softmax_output: the softmax values from forward pass
    """
    s = softmax_output
    grad_input = s * grad_output  # Element-wise
    grad_input -= s * np.sum(grad_output * s)  # Subtract weighted sum
    return grad_input

# Test softmax backward
scores = np.array([1.0, 2.0, 3.0])
probs = softmax(scores)
print(f"Softmax output: {probs}")

# Say we want the first probability to increase
grad_out = np.array([1.0, 0.0, 0.0])  
grad_scores = softmax_backward(grad_out, probs)
print(f"Gradient to scores: {grad_scores}")
print(f"Sum of gradients: {grad_scores.sum()}")  # Should be ~0!
class FeedForward:
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Xavier initialization
        self.W1 = np.random.randn(d_model, self.d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * np.sqrt(2.0 / self.d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        self.x = x  
        self.z1 = self.x @ self.W1 + self.b1  # Fixed: self.b1
        self.hidden = np.maximum(0, self.z1)  
        return self.hidden @ self.W2 + self.b2  # Made it code, not comment
    
    def backward(self, grad_output):
        self.grad_W2 = self.hidden.T @ grad_output  # Fixed: self.hidden
        self.grad_b2 = grad_output.sum(axis=0)
        
        grad_hidden = grad_output @ self.W2.T
        grad_z1 = grad_hidden * (self.z1 > 0)
        
        self.grad_W1 = self.x.T @ grad_z1
        self.grad_b1 = grad_z1.sum(axis=0)  # Fixed: axis not dim
        
        return grad_z1 @ self.W1.
class FeedForward:
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Xavier initialization
        self.W1 = np.random.randn(d_model, self.d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * np.sqrt(2.0 / self.d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        self.x = x  
        self.z1 = self.x @ self.W1 + self.b1  # Fixed: self.b1
        self.hidden = np.maximum(0, self.z1)  
        return self.hidden @ self.W2 + self.b2  # Made it code, not comment
    
    def backward(self, grad_output):
        self.grad_W2 = self.hidden.T @ grad_output  # Fixed: self.hidden
        self.grad_b2 = grad_output.sum(axis=0)
        
        grad_hidden = grad_output @ self.W2.T
        grad_z1 = grad_hidden * (self.z1 > 0)
        
        self.grad_W1 = self.x.T @ grad_z1
        self.grad_b1 = grad_z1.sum(axis=0)  # Fixed: axis not dim
        
        return grad_z1 @ self.W1.T  # Return gradient for previous layer
%save session_aug_18_2025.py 1-
%save session_aug_18_2025.py 1
%save session_aug_18_2025.py 1-
%history 1- -f session_aug_18_2025.py
%history 1- -f session_aug_18_2025.py
# See how big the file is
import os
os.path.getsize('session_aug_18_2025.py')

# Or peek at the last few lines to make sure it got your recent work
with open('session_aug_18_2025.py', 'r') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print("Last 10 lines:")
    print(''.join(lines[-10:]))
%history
%history -f session_backuup_18_aug_2025.py
