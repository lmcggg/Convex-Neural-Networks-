"""
Data generation for test systems:
1. Delay system (sanity check from paper §4.2.3)
2. Nonlinear NARX system (main evaluation)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def generate_delay_system(n_samples=50000, seed=42):
    """
    Delay system from paper §4.2.3:
        u_t ~ Uniform[-1, 1]
        y_t = u_{t-4} + u_{t-3} + u_{t-2} + u_{t-1}
    
    Returns:
        u: (n_samples,) input sequence
        y: (n_samples,) output sequence
    """
    np.random.seed(seed)
    
    # Generate input
    u = np.random.uniform(-1, 1, size=n_samples + 4)
    
    # Generate output with delay
    y = np.zeros(n_samples)
    for t in range(n_samples):
        y[t] = u[t] + u[t+1] + u[t+2] + u[t+3]
    
    # Trim input to match output length
    u = u[4:]
    
    return u, y


def generate_nonlinear_narx(n_samples=50000, noise_std=0.01, seed=42):
    """
    Nonlinear NARX system:
        u_t ~ Uniform[-1, 1]
        y_t = 0.25*y_{t-1} + 0.5*tanh(0.8*y_{t-2} + 0.8*u_{t-1}) 
              + 0.15*u_{t-2}^2 - 0.1*u_{t-3}*u_{t-2} + ε_t
        ε_t ~ N(0, 0.01^2)
    
    Returns:
        u: (n_samples,) input sequence
        y: (n_samples,) output sequence
    """
    np.random.seed(seed)
    
    # Generate input
    u = np.random.uniform(-1, 1, size=n_samples + 10)  # Extra for warmup
    
    # Initialize output
    y = np.zeros(n_samples + 10)
    
    # Warmup period
    for t in range(10):
        noise = np.random.normal(0, noise_std)
        y[t] = noise
    
    # Generate output sequence
    for t in range(10, n_samples + 10):
        y_t_1 = y[t-1]
        y_t_2 = y[t-2]
        u_t_1 = u[t-1]
        u_t_2 = u[t-2]
        u_t_3 = u[t-3]
        
        noise = np.random.normal(0, noise_std)
        
        y[t] = (0.25 * y_t_1 + 
                0.5 * np.tanh(0.8 * y_t_2 + 0.8 * u_t_1) + 
                0.15 * u_t_2**2 - 
                0.1 * u_t_3 * u_t_2 + 
                noise)
    
    # Remove warmup period
    u = u[10:]
    y = y[10:]
    
    return u, y


class SequenceDataset(Dataset):
    """
    Dataset for sequence-to-sequence learning with sliding windows
    """
    def __init__(self, u, y, window_size=5, include_output_history=False):
        """
        Args:
            u: (n_samples,) input sequence
            y: (n_samples,) output sequence
            window_size: number of time steps in input window
            include_output_history: whether to include y_{t-2}, y_{t-1} in input
        """
        self.u = torch.FloatTensor(u)
        self.y = torch.FloatTensor(y)
        self.window_size = window_size
        self.include_output_history = include_output_history
        
        # Compute valid indices
        self.valid_indices = list(range(window_size, len(u)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # Input window: [u_{t-window_size+1}, ..., u_t]
        u_window = self.u[t-self.window_size+1:t+1].unsqueeze(-1)  # (window_size, 1)
        
        if self.include_output_history and t >= 2:
            # Add y_{t-2}, y_{t-1} to the last time step
            y_hist = torch.stack([self.y[t-2], self.y[t-1]])  # (2,)
            # For simplicity, we'll return them separately
            return u_window, self.y[t], y_hist
        else:
            return u_window, self.y[t]


def create_dataloaders(u, y, window_size=5, batch_size=32, 
                       train_ratio=0.7, val_ratio=0.15, 
                       include_output_history=False):
    """
    Create train/val/test dataloaders with proper normalization
    
    Returns:
        train_loader, val_loader, test_loader, u_stats, y_stats
    """
    n_samples = len(u)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split data
    u_train = u[:n_train]
    y_train = y[:n_train]
    
    u_val = u[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    
    u_test = u[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    # Compute normalization statistics from training set
    u_mean, u_std = np.mean(u_train), np.std(u_train)
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    
    # Normalize
    u_train_norm = (u_train - u_mean) / (u_std + 1e-8)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    
    u_val_norm = (u_val - u_mean) / (u_std + 1e-8)
    y_val_norm = (y_val - y_mean) / (y_std + 1e-8)
    
    u_test_norm = (u_test - u_mean) / (u_std + 1e-8)
    y_test_norm = (y_test - y_mean) / (y_std + 1e-8)
    
    # Create datasets
    train_dataset = SequenceDataset(u_train_norm, y_train_norm, window_size, include_output_history)
    val_dataset = SequenceDataset(u_val_norm, y_val_norm, window_size, include_output_history)
    test_dataset = SequenceDataset(u_test_norm, y_test_norm, window_size, include_output_history)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    u_stats = {'mean': u_mean, 'std': u_std}
    y_stats = {'mean': y_mean, 'std': y_std}
    
    return train_loader, val_loader, test_loader, u_stats, y_stats


def compute_rollout_mse(model, u_seq, y_seq, window_size, rollout_steps, 
                        u_stats, y_stats, device='cpu'):
    """
    Compute K-step rollout MSE on test sequence
    
    Args:
        model: trained RNN model
        u_seq: raw input sequence (not normalized)
        y_seq: raw output sequence (not normalized)
        window_size: input window size
        rollout_steps: number of steps to rollout
        u_stats, y_stats: normalization statistics
        device: torch device
    
    Returns:
        mse: mean squared error over rollout
    """
    model.eval()
    
    # Normalize
    u_norm = (u_seq - u_stats['mean']) / (u_stats['std'] + 1e-8)
    y_norm = (y_seq - y_stats['mean']) / (y_stats['std'] + 1e-8)
    
    # Convert to tensors
    u_tensor = torch.FloatTensor(u_norm).to(device)
    y_tensor = torch.FloatTensor(y_norm).to(device)
    
    # Select a starting point with enough history
    start_idx = window_size
    end_idx = start_idx + rollout_steps
    
    if end_idx > len(u_seq):
        end_idx = len(u_seq)
        rollout_steps = end_idx - start_idx
    
    # Prepare initial input window
    u_window = u_tensor[start_idx-window_size:end_idx].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
    
    with torch.no_grad():
        y_pred_norm = model(u_window)  # (1, seq_len, 1)
    
    # Denormalize predictions
    y_pred = y_pred_norm.squeeze().cpu().numpy() * y_stats['std'] + y_stats['mean']
    y_true = y_seq[start_idx:end_idx]
    
    mse = np.mean((y_pred - y_true)**2)
    return mse

