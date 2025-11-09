# Convex Neural Networks Implementation

Complete implementation of Input Convex Neural Networks (ICNN/ICRNN) and Convex Difference Neural Networks (CDiNN) for system identification.

## Networks Implemented

### 1. ICNN (Input Convex Neural Network)
Feedforward network that is convex with respect to its input through:
- Input expansion: û = [u; -u]
- Non-negative weight constraints on layer-to-layer connections
- Convex monotone activations (ReLU)

### 2. ICRNN (Input Convex Recurrent Neural Network)
Recurrent version with equations:
```
z_t = σ₁(U û_t + W z_{t-1} + D₂ û_{t-1})
y_t = σ₂(V z_t + D₁ z_{t-1} + D₃ û_t)
```
All weights U, V, W, D₁, D₂, D₃ ≥ 0

### 3. CDiNN-1 (Convex Difference Network - Type 1)
ICNN backbone with unconstrained output layer, representing output as difference of convex functions.

### 4. CDiNN-2 (Convex Difference Network - Type 2)
Two parallel ICNNs with output y = f₁(x) - f₂(x).

### 5. R-CDiNN (Recurrent CDiNN)
Recurrent architecture with:
```
z_t = PC-ReLU(U x_t + Z z_{t-1})
y_t = M z_t
```
Where Z ≥ 0 (convexity) and M is unconstrained (DC representation).

## Key Features

- **PC-ReLU Activation**: Parametrically Constrained ReLU with learnable parameter a ∈ [0,1]
- **Non-negative Weight Constraints**: Implemented via softplus parameterization
- **Bias-free Training**: Following paper recommendations for better generalization
- **Input Expansion**: û = [u; -u] to express negative relationships under non-negative constraints

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- NumPy
- Matplotlib (for visualizations)

## Usage

Run the complete demonstration on both test systems:

```bash
python demo.py
```

This will:
1. Test on delay system (sanity check): y_t = u_{t-4} + u_{t-3} + u_{t-2} + u_{t-1}
2. Test on nonlinear NARX system with saturation, cross terms, and delays
3. Train and evaluate all 5 network architectures
4. Report 1-step MSE and 50-step rollout MSE
5. Generate visualization plots automatically



## Project Structure

```
.
├── networks.py              # All network architectures
├── data_generation.py       # Test system data generation
├── train_utils.py           # Training and evaluation utilities
├── visualize.py             # Visualization utilities
├── demo.py                  # Demonstration script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Test Systems

### Delay System (Sanity Check)
```
u_t ~ Uniform[-1, 1]
y_t = u_{t-4} + u_{t-3} + u_{t-2} + u_{t-1}
```

### Nonlinear NARX System (Main Evaluation)
```
u_t ~ Uniform[-1, 1]
y_t = 0.25·y_{t-1} + 0.5·tanh(0.8·y_{t-2} + 0.8·u_{t-1}) 
      + 0.15·u_{t-2}² - 0.1·u_{t-3}·u_{t-2} + ε_t
ε_t ~ N(0, 0.01²)
```

## Implementation Details

### Non-negative Constraints
Weights requiring non-negativity use softplus parameterization:
```python
W = softplus(Θ)
```

### PC-ReLU Parameter
The parameter a is constrained to [0,1] using sigmoid:
```python
a = sigmoid(ã)
```

### Training
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam
- Early stopping with patience
- Normalization: Zero mean, unit variance on training set

## References

Based on specifications from:
- ICNN/ICRNN papers on input convex networks
- CDiNN paper on convex difference neural networks
- Implementation follows nn.md specifications exactly

## License

MIT


