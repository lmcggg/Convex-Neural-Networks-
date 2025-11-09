"""
Training and evaluation utilities for all network types
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train_rnn_model(model, train_loader, val_loader, epochs=100, lr=1e-3, 
                    device='cpu', patience=15, verbose=True, clip_grad=1.0):
    """
    Train recurrent models (ICRNN, R-CDiNN)
    
    Args:
        model: RNN model
        train_loader: training dataloader
        val_loader: validation dataloader
        epochs: number of training epochs
        lr: learning rate
        device: torch device
        patience: early stopping patience
        verbose: whether to print progress
        clip_grad: gradient clipping value (None to disable)
    
    Returns:
        model: trained model
        history: training history dict
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            u_window, y_target = batch
            u_window = u_window.to(device)  # (batch, window_size, 1)
            y_target = y_target.to(device).unsqueeze(-1)  # (batch, 1)
            
            optimizer.zero_grad()
            
            # Forward pass - predict only the last time step
            y_pred_seq = model(u_window)  # (batch, window_size, 1)
            y_pred = y_pred_seq[:, -1, :]  # (batch, 1) - last time step
            
            loss = criterion(y_pred, y_target)
            loss.backward()
            
            # Gradient clipping for stability
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                u_window, y_target = batch
                u_window = u_window.to(device)
                y_target = y_target.to(device).unsqueeze(-1)
                
                y_pred_seq = model(u_window)
                y_pred = y_pred_seq[:, -1, :]
                
                loss = criterion(y_pred, y_target)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    return model, history


def evaluate_rnn_model(model, test_loader, device='cpu'):
    """
    Evaluate RNN model on test set (1-step prediction)
    
    Returns:
        mse: mean squared error
    """
    model.eval()
    criterion = nn.MSELoss()
    test_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            u_window, y_target = batch
            u_window = u_window.to(device)
            y_target = y_target.to(device).unsqueeze(-1)
            
            y_pred_seq = model(u_window)
            y_pred = y_pred_seq[:, -1, :]
            
            loss = criterion(y_pred, y_target)
            test_losses.append(loss.item())
    
    return np.mean(test_losses)


def train_feedforward_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                            device='cpu', patience=15, verbose=True):
    """
    Train feedforward models (ICNN, CDiNN-1, CDiNN-2)
    
    Note: For feedforward models, we need to flatten the sequence window
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            u_window, y_target = batch
            # Flatten window: (batch, window_size, 1) -> (batch, window_size)
            u_flat = u_window.squeeze(-1).to(device)
            y_target = y_target.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            
            y_pred = model(u_flat)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                u_window, y_target = batch
                u_flat = u_window.squeeze(-1).to(device)
                y_target = y_target.to(device).unsqueeze(-1)
                
                y_pred = model(u_flat)
                loss = criterion(y_pred, y_target)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    return model, history


def evaluate_feedforward_model(model, test_loader, device='cpu'):
    """
    Evaluate feedforward model on test set
    
    Returns:
        mse: mean squared error
    """
    model.eval()
    criterion = nn.MSELoss()
    test_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            u_window, y_target = batch
            u_flat = u_window.squeeze(-1).to(device)
            y_target = y_target.to(device).unsqueeze(-1)
            
            y_pred = model(u_flat)
            loss = criterion(y_pred, y_target)
            test_losses.append(loss.item())
    
    return np.mean(test_losses)


def compute_rollout_mse_rnn(model, u_seq, y_seq, window_size, rollout_steps,
                            u_stats, y_stats, device='cpu'):
    """
    Compute K-step rollout MSE for RNN models
    
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
    
    # Select a starting point with enough history
    start_idx = window_size
    end_idx = min(start_idx + rollout_steps, len(u_seq))
    actual_rollout = end_idx - start_idx
    
    # Prepare input sequence (including window_size history + rollout_steps)
    u_window = u_tensor[start_idx-window_size:end_idx].unsqueeze(0).unsqueeze(-1)  # (1, window_size+rollout, 1)
    
    with torch.no_grad():
        y_pred_norm = model(u_window)  # (1, window_size+rollout, 1)
    
    # Take only the last 'actual_rollout' predictions (corresponding to start_idx:end_idx)
    y_pred_norm = y_pred_norm[:, -actual_rollout:, :]
    
    # Denormalize predictions
    y_pred = y_pred_norm.squeeze().cpu().numpy() * y_stats['std'] + y_stats['mean']
    y_true = y_seq[start_idx:end_idx]
    
    # Handle single prediction case
    if y_pred.ndim == 0:
        y_pred = np.array([y_pred])
    
    # Check for numerical issues
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return float('nan')
    
    mse = np.mean((y_pred - y_true)**2)
    return mse


def verify_non_negative_weights(model, verbose=True):
    """
    Verify that non-negative weight constraints are satisfied
    
    Returns:
        all_satisfied: bool
        violations: list of (name, min_value) tuples for violations
    """
    violations = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and '_weight_raw' in name:
            # This is a non-negative weight (before softplus)
            actual_weight = torch.nn.functional.softplus(param)
            min_val = actual_weight.min().item()
            if min_val < -1e-6:  # Small tolerance for numerical errors
                violations.append((name, min_val))
    
    all_satisfied = len(violations) == 0
    
    if verbose:
        if all_satisfied:
            print("[OK] All non-negative weight constraints satisfied")
        else:
            print("[FAIL] Non-negative weight constraint violations:")
            for name, min_val in violations:
                print(f"  {name}: min = {min_val}")
    
    return all_satisfied, violations

