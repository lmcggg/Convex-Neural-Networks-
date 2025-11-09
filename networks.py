"""
Complete implementation of ICNN, ICRNN, CDiNN, and R-CDiNN networks.
Based on the specifications in nn.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCReLU(nn.Module):
    """Parametrically Constrained ReLU: PC-ReLU(x) = max(a*x, x), a ∈ [0,1]
    
    Each hidden unit has its own parameter a_i ∈ [0,1]
    """
    def __init__(self, size):
        super().__init__()
        # Each element has its own a parameter, initialized to give a≈0.5
        self._a_logit = nn.Parameter(torch.zeros(size))
    
    @property
    def a(self):
        return torch.sigmoid(self._a_logit)
    
    def forward(self, x):
        # a is broadcast to match x's batch dimension
        return torch.maximum(self.a * x, x)


class NonNegativeLinear(nn.Module):
    """Linear layer with non-negative weights using softplus parameterization"""
    def __init__(self, in_features, out_features, bias=True, init_scale=0.1):
        super().__init__()
        # Initialize small for stability: softplus(-2) ≈ 0.13
        self._weight_raw = nn.Parameter(torch.randn(out_features, in_features) * init_scale - 2.0)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    
    @property
    def weight(self):
        return F.softplus(self._weight_raw)
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out


class NonNegRowNormLinear(nn.Module):
    """
    Non-negative weights + row L1 normalization ≤ c for contractive recurrent matrix Z.
    Ensures both non-negativity and contraction (row sum ≤ c where c < 1).
    """
    def __init__(self, in_features, out_features, bias=False, c=0.95, init_scale=0.1):
        super().__init__()
        # softplus(-2) ≈ 0.13, small initialization for stability
        self._weight_raw = nn.Parameter(
            torch.randn(out_features, in_features) * init_scale - 2.0
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.c = c
    
    @property
    def weight(self):
        W = F.softplus(self._weight_raw)                 # Ensure non-negativity
        row_sum = W.sum(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(self.c / row_sum, max=1.0)   # Row sum ≤ c
        return W * scale
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class ICNN(nn.Module):
    """
    Input Convex Neural Network (Feedforward)
    
    Architecture:
        z_1 = σ(W_1 û + b_1)
        z_k = σ(W_k z_{k-1} + D_k û + b_k), k=2,...,K
        y = W_{K+1} z_K + D_{K+1} û
    
    Constraints: W_{1:K+1} ≥ 0, D_{2:K+1} ≥ 0, σ convex and monotone non-decreasing
    """
    def __init__(self, input_dim, hidden_dims, use_bias=True, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expanded_dim = 2 * input_dim  # û = [u; -u]
        
        layers = []
        passthrough_layers = []
        
        # First layer: W_1 û + b_1
        layers.append(NonNegativeLinear(self.expanded_dim, hidden_dims[0], bias=use_bias))
        passthrough_layers.append(None)
        
        # Hidden layers: W_k z_{k-1} + D_k û + b_k
        for k in range(1, len(hidden_dims)):
            layers.append(NonNegativeLinear(hidden_dims[k-1], hidden_dims[k], bias=use_bias))
            passthrough_layers.append(NonNegativeLinear(self.expanded_dim, hidden_dims[k], bias=False))
        
        self.layers = nn.ModuleList(layers)
        self.passthrough_layers = nn.ModuleList(passthrough_layers)
        self.activation = nn.ReLU()  # Convex and monotone non-decreasing
        
        # Output layer: W_{K+1} z_K + D_{K+1} û (maintains convexity)
        self.output_layer = NonNegativeLinear(hidden_dims[-1], output_dim, bias=use_bias)
        self.output_passthrough = NonNegativeLinear(self.expanded_dim, output_dim, bias=False)
    
    def expand_input(self, u):
        """Expand input: û = [u; -u]"""
        return torch.cat([u, -u], dim=-1)
    
    def forward(self, u):
        u_hat = self.expand_input(u)
        
        # First layer
        z = self.activation(self.layers[0](u_hat))
        
        # Hidden layers with passthrough
        for k in range(1, len(self.layers)):
            z = self.layers[k](z) + self.passthrough_layers[k](u_hat)
            z = self.activation(z)
        
        # Output layer (no activation to allow full range)
        y = self.output_layer(z) + self.output_passthrough(u_hat)
        return y


class ICRNN(nn.Module):
    """
    Input Convex Recurrent Neural Network
    
    Forward equations (paper Eq. 2-3):
        z_t = σ_1(U û_t + W z_{t-1} + D_2 û_{t-1})
        y_t = σ_2(V z_t + D_1 z_{t-1} + D_3 û_t)
    
    Constraints: U, V, W, D_1, D_2, D_3 ≥ 0, σ_1, σ_2 convex and monotone
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.expanded_dim = 2 * input_dim
        
        # All weights must be non-negative
        self.U = NonNegativeLinear(self.expanded_dim, hidden_dim, bias=use_bias)
        self.W = NonNegativeLinear(hidden_dim, hidden_dim, bias=False)
        self.D2 = NonNegativeLinear(self.expanded_dim, hidden_dim, bias=False)
        
        self.V = NonNegativeLinear(hidden_dim, output_dim, bias=use_bias)
        self.D1 = NonNegativeLinear(hidden_dim, output_dim, bias=False)
        self.D3 = NonNegativeLinear(self.expanded_dim, output_dim, bias=False)
        
        self.sigma1 = nn.ReLU()
        self.sigma2 = nn.ReLU()
    
    def expand_input(self, u):
        """Expand input: û = [u; -u]"""
        return torch.cat([u, -u], dim=-1)
    
    def forward(self, u_seq, z_init=None):
        """
        Args:
            u_seq: (batch, seq_len, input_dim)
            z_init: (batch, hidden_dim) or None
        Returns:
            y_seq: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = u_seq.shape
        
        if z_init is None:
            z_prev = torch.zeros(batch_size, self.hidden_dim, device=u_seq.device)
        else:
            z_prev = z_init
        
        u_hat_prev = torch.zeros(batch_size, self.expanded_dim, device=u_seq.device)
        
        outputs = []
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            u_hat_t = self.expand_input(u_t)
            
            # z_t = σ_1(U û_t + W z_{t-1} + D_2 û_{t-1})
            z_t = self.sigma1(
                self.U(u_hat_t) + self.W(z_prev) + self.D2(u_hat_prev)
            )
            
            # y_t = σ_2(V z_t + D_1 z_{t-1} + D_3 û_t)
            y_t = self.sigma2(
                self.V(z_t) + self.D1(z_prev) + self.D3(u_hat_t)
            )
            
            outputs.append(y_t)
            z_prev = z_t
            u_hat_prev = u_hat_t
        
        return torch.stack(outputs, dim=1)


class CDiNN1(nn.Module):
    """
    CDiNN-1: ICNN backbone + unconstrained output layer
    
    Structure: ICNN with all layers non-negative except W_0^(x) and W_K^(z)
    Output: y = W_K^(z) z_K (difference of convex functions)
    """
    def __init__(self, input_dim, hidden_dims, output_dim, use_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.expanded_dim = 2 * input_dim
        
        layers = []
        passthrough_layers = []
        activations = []
        
        # First layer: no constraint on W_0^(x)
        layers.append(nn.Linear(self.expanded_dim, hidden_dims[0], bias=use_bias))
        passthrough_layers.append(None)
        activations.append(PCReLU(hidden_dims[0]))
        
        # Hidden layers: W_k ≥ 0, D_k ≥ 0
        for k in range(1, len(hidden_dims)):
            layers.append(NonNegativeLinear(hidden_dims[k-1], hidden_dims[k], bias=use_bias))
            passthrough_layers.append(NonNegativeLinear(self.expanded_dim, hidden_dims[k], bias=False))
            activations.append(PCReLU(hidden_dims[k]))
        
        self.layers = nn.ModuleList(layers)
        self.passthrough_layers = nn.ModuleList(passthrough_layers)
        self.activations = nn.ModuleList(activations)
        
        # Output layer: no non-negative constraint (allows DC representation)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim, bias=use_bias)
    
    def expand_input(self, x):
        """Expand input: x̂ = [x; -x]"""
        return torch.cat([x, -x], dim=-1)
    
    def forward(self, x):
        x_hat = self.expand_input(x)
        
        # First layer
        z = self.activations[0](self.layers[0](x_hat))
        
        # Hidden layers with passthrough
        for k in range(1, len(self.layers)):
            z = self.layers[k](z) + self.passthrough_layers[k](x_hat)
            z = self.activations[k](z)
        
        # Unconstrained output layer
        y = self.output_layer(z)
        return y


class CDiNN2(nn.Module):
    """
    CDiNN-2: Two parallel ICNNs with difference output
    
    Structure: y = f_1(x) - f_2(x), where f_1 and f_2 are ICNNs
    """
    def __init__(self, input_dim, hidden_dims, output_dim, use_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.expanded_dim = 2 * input_dim
        
        # Two independent ICNN branches
        self.icnn1 = self._build_icnn(hidden_dims, output_dim, use_bias)
        self.icnn2 = self._build_icnn(hidden_dims, output_dim, use_bias)
    
    def _build_icnn(self, hidden_dims, output_dim, use_bias):
        """Build a single ICNN branch"""
        layers = []
        passthrough_layers = []
        activations = []
        
        # First layer: no constraint on W_0^(x)
        layers.append(nn.Linear(self.expanded_dim, hidden_dims[0], bias=use_bias))
        passthrough_layers.append(None)
        activations.append(PCReLU(hidden_dims[0]))
        
        # Hidden layers: W_k ≥ 0, D_k ≥ 0
        for k in range(1, len(hidden_dims)):
            layers.append(NonNegativeLinear(hidden_dims[k-1], hidden_dims[k], bias=use_bias))
            passthrough_layers.append(NonNegativeLinear(self.expanded_dim, hidden_dims[k], bias=False))
            activations.append(PCReLU(hidden_dims[k]))
        
        # Output layer: non-negative for convexity
        layers.append(NonNegativeLinear(hidden_dims[-1], output_dim, bias=use_bias))
        passthrough_layers.append(None)
        activations.append(None)
        
        return nn.ModuleDict({
            'layers': nn.ModuleList(layers),
            'passthrough_layers': nn.ModuleList(passthrough_layers),
            'activations': nn.ModuleList(activations)
        })
    
    def expand_input(self, x):
        """Expand input: x̂ = [x; -x]"""
        return torch.cat([x, -x], dim=-1)
    
    def _forward_icnn(self, x_hat, icnn):
        """Forward pass through a single ICNN"""
        layers = icnn['layers']
        passthrough_layers = icnn['passthrough_layers']
        activations = icnn['activations']
        
        # First layer
        z = activations[0](layers[0](x_hat))
        
        # Hidden layers with passthrough
        for k in range(1, len(layers) - 1):
            z = layers[k](z) + passthrough_layers[k](x_hat)
            z = activations[k](z)
        
        # Output layer
        z = layers[-1](z)
        return z
    
    def forward(self, x):
        x_hat = self.expand_input(x)
        
        # Forward through both ICNNs
        f1 = self._forward_icnn(x_hat, self.icnn1)
        f2 = self._forward_icnn(x_hat, self.icnn2)
        
        # Difference of convex functions
        y = f1 - f2
        return y


class RCDiNN(nn.Module):
    """
    Recurrent CDiNN (R-CDiNN)
    
    Forward equations (paper Eq. 10):
        z_t = PC-ReLU(U x_t + Z z_{t-1})
        y_t = M z_t
    
    Constraints: Z ≥ 0 with row sum ≤ c (ensures convexity + contraction), 
                 M unconstrained (allows DC)
    Recommendation: disable bias
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # U: no constraint, small initialization for stability
        self.U = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.U.weight, gain=0.5)
        
        # Z: non-negative + contractive constraint (row sum ≤ 0.95)
        self.Z = NonNegRowNormLinear(hidden_dim, hidden_dim, bias=False, c=0.95, init_scale=0.1)
        
        # M: no constraint (allows DC representation), small initialization
        self.M = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.M.weight, gain=0.5)
        
        # PC-ReLU with per-unit parameters
        self.activation = PCReLU(hidden_dim)
    
    def forward(self, x_seq, z_init=None):
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
            z_init: (batch, hidden_dim) or None
        Returns:
            y_seq: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x_seq.shape
        
        if z_init is None:
            z_prev = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        else:
            z_prev = z_init
        
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            # z_t = PC-ReLU(U x_t + Z z_{t-1})
            z_t = self.activation(self.U(x_t) + self.Z(z_prev))
            
            # y_t = M z_t
            y_t = self.M(z_t)
            
            outputs.append(y_t)
            z_prev = z_t
        
        return torch.stack(outputs, dim=1)

