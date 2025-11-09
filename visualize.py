"""
Visualization utilities for comparing model predictions with actual system outputs
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_predictions(model, u_seq, y_true, window_size, u_stats, y_stats, 
                          rollout_steps=50, device='cpu', model_name='Model',
                          save_path=None):
    """
    Visualize model predictions vs actual outputs for both 1-step and multi-step rollout
    
    Args:
        model: Trained RNN model (RCDiNN, ICRNN, etc.)
        u_seq: Input sequence (n_samples,)
        y_true: True output sequence (n_samples,)
        window_size: Window size for initial context
        u_stats: dict with 'mean' and 'std' for input normalization
        y_stats: dict with 'mean' and 'std' for output normalization
        rollout_steps: Number of steps for rollout prediction
        device: 'cpu' or 'cuda'
        model_name: Name of the model for plot title
        save_path: Path to save the figure (optional)
    """
    model.eval()
    model.to(device)
    
    u_mean = u_stats['mean']
    u_std = u_stats['std']
    y_mean = y_stats['mean']
    y_std = y_stats['std']
    
    # Normalize inputs
    u_norm = (u_seq - u_mean) / (u_std + 1e-8)
    y_norm = (y_true - y_mean) / (y_std + 1e-8)
    
    # Select a segment for visualization (avoid the very beginning)
    start_idx = 100
    vis_length = min(200, len(u_seq) - start_idx - rollout_steps)
    
    # === 1-step prediction ===
    y_pred_1step = []
    
    with torch.no_grad():
        for t in range(start_idx, start_idx + vis_length):
            # Use window of past inputs (teacher forcing)
            if t < window_size:
                # Not enough history, skip
                continue
            
            u_window = u_norm[t-window_size:t]  # Get window_size past inputs
            u_tensor = torch.FloatTensor(u_window).unsqueeze(0).unsqueeze(-1).to(device)  # (1, window_size, 1)
            
            # Predict next step
            y_pred_seq = model(u_tensor)  # (1, window_size, 1)
            y_pred = y_pred_seq[:, -1, :]  # Take last prediction
            y_pred_1step.append(y_pred[0, 0].cpu().item())
    
    y_pred_1step = np.array(y_pred_1step)
    
    # Denormalize
    y_pred_1step_denorm = y_pred_1step * (y_std + 1e-8) + y_mean
    # Adjust indices to match predictions (we start from start_idx but predictions start from window_size if needed)
    actual_start = max(start_idx, window_size)
    y_true_segment = y_true[actual_start:actual_start + len(y_pred_1step)]
    u_segment = u_seq[actual_start:actual_start + len(y_pred_1step)]
    time_axis = np.arange(actual_start, actual_start + len(y_pred_1step))
    
    # === Multi-step rollout ===
    rollout_start = actual_start + len(y_pred_1step) // 2
    y_pred_rollout = []
    
    with torch.no_grad():
        # For rollout, we feed the entire input sequence at once
        # The model processes all timesteps and we take the predictions
        rollout_end = min(rollout_start + rollout_steps, len(u_norm))
        actual_rollout_steps = rollout_end - rollout_start
        
        # Get input window including history
        u_rollout_window = u_norm[rollout_start-window_size:rollout_end]
        u_tensor = torch.FloatTensor(u_rollout_window).unsqueeze(0).unsqueeze(-1).to(device)  # (1, window_size+rollout_steps, 1)
        
        # Get predictions for all timesteps
        y_pred_seq = model(u_tensor)  # (1, window_size+rollout_steps, 1)
        
        # Extract only the rollout predictions (last actual_rollout_steps)
        y_pred_rollout = y_pred_seq[0, -actual_rollout_steps:, 0].cpu().numpy()
    
    y_pred_rollout_denorm = y_pred_rollout * (y_std + 1e-8) + y_mean
    y_true_rollout = y_true[rollout_start:rollout_start + actual_rollout_steps]
    
    # === Create visualization ===
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Input sequence
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, u_segment, 'b-', linewidth=1, alpha=0.7)
    ax1.axvline(x=rollout_start, color='r', linestyle='--', alpha=0.5, label='Rollout Start')
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Input u(t)', fontsize=11)
    ax1.set_title(f'{model_name} - Input Sequence', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: 1-step prediction (full view)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_axis, y_true_segment, 'g-', linewidth=2, label='True Output', alpha=0.8)
    ax2.plot(time_axis, y_pred_1step_denorm, 'r--', linewidth=1.5, label='Predicted Output', alpha=0.8)
    ax2.axvline(x=rollout_start, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Output y(t)', fontsize=11)
    ax2.set_title(f'{model_name} - 1-Step Prediction', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Calculate 1-step MSE
    mse_1step = np.mean((y_true_segment - y_pred_1step_denorm) ** 2)
    ax2.text(0.02, 0.98, f'MSE: {mse_1step:.6f}', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Multi-step rollout
    ax3 = fig.add_subplot(gs[2, 0])
    rollout_time = np.arange(rollout_start, rollout_start + actual_rollout_steps)
    ax3.plot(rollout_time, y_true_rollout, 'g-', linewidth=2, label='True Output', alpha=0.8)
    ax3.plot(rollout_time, y_pred_rollout_denorm, 'r--', linewidth=1.5, label='Predicted Output', alpha=0.8)
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Output y(t)', fontsize=11)
    ax3.set_title(f'{model_name} - {rollout_steps}-Step Rollout', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Calculate rollout MSE
    mse_rollout = np.mean((y_true_rollout - y_pred_rollout_denorm) ** 2)
    ax3.text(0.02, 0.98, f'MSE: {mse_rollout:.6f}', transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Prediction error over time (rollout)
    ax4 = fig.add_subplot(gs[2, 1])
    error = np.abs(y_true_rollout - y_pred_rollout_denorm)
    ax4.plot(rollout_time, error, 'purple', linewidth=2, alpha=0.7)
    ax4.fill_between(rollout_time, 0, error, alpha=0.3, color='purple')
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Absolute Error', fontsize=11)
    ax4.set_title('Rollout Prediction Error', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add mean and max error annotations
    mean_error = np.mean(error)
    max_error = np.max(error)
    ax4.axhline(y=mean_error, color='orange', linestyle='--', linewidth=1, label=f'Mean: {mean_error:.4f}')
    ax4.text(0.02, 0.98, f'Max Error: {max_error:.4f}', transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.legend(fontsize=9)
    
    plt.suptitle(f'{model_name} - Prediction Visualization', fontsize=15, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    return fig


def visualize_rollout_comparison(models_dict, u_seq, y_true, window_size, 
                                 u_stats, y_stats, rollout_steps=50, 
                                 device='cpu', save_path=None):
    """
    Compare multiple models' rollout predictions side by side
    
    Args:
        models_dict: Dictionary of {model_name: model}
        u_seq: Input sequence (n_samples,)
        y_true: True output sequence (n_samples,)
        window_size: Window size for initial context
        u_stats: dict with 'mean' and 'std' for input normalization
        y_stats: dict with 'mean' and 'std' for output normalization
        rollout_steps: Number of steps for rollout prediction
        device: 'cpu' or 'cuda'
        save_path: Path to save the figure (optional)
    """
    u_mean = u_stats['mean']
    u_std = u_stats['std']
    y_mean = y_stats['mean']
    y_std = y_stats['std']
    
    # Normalize
    u_norm = (u_seq - u_mean) / (u_std + 1e-8)
    
    # Select rollout segment
    rollout_start = max(500, window_size)
    rollout_end = min(rollout_start + rollout_steps, len(u_norm))
    actual_rollout_steps = rollout_end - rollout_start
    
    # Collect predictions from all models
    predictions = {}
    mse_values = {}
    
    for model_name, model in models_dict.items():
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            # Get input window including history
            u_rollout_window = u_norm[rollout_start-window_size:rollout_end]
            u_tensor = torch.FloatTensor(u_rollout_window).unsqueeze(0).unsqueeze(-1).to(device)  # (1, window_size+rollout_steps, 1)
            
            # Get predictions for all timesteps
            y_pred_seq = model(u_tensor)  # (1, window_size+rollout_steps, 1)
            
            # Extract only the rollout predictions (last actual_rollout_steps)
            y_pred_rollout = y_pred_seq[0, -actual_rollout_steps:, 0].cpu().numpy()
        
        y_pred_denorm = y_pred_rollout * (y_std + 1e-8) + y_mean
        
        predictions[model_name] = y_pred_denorm
        mse_values[model_name] = np.mean((y_true[rollout_start:rollout_start + actual_rollout_steps] - y_pred_denorm) ** 2)
    
    # Create comparison plot
    fig, axes = plt.subplots(len(models_dict) + 1, 1, figsize=(14, 4 * (len(models_dict) + 1)))
    
    if len(models_dict) == 1:
        axes = [axes]
    
    rollout_time = np.arange(rollout_start, rollout_start + actual_rollout_steps)
    y_true_rollout = y_true[rollout_start:rollout_start + actual_rollout_steps]
    
    # Plot input
    axes[0].plot(rollout_time, u_seq[rollout_start:rollout_start + actual_rollout_steps], 
                 'b-', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Input u(t)', fontsize=11)
    axes[0].set_title('Input Sequence', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each model's prediction
    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx + 1]
        
        ax.plot(rollout_time, y_true_rollout, 'g-', linewidth=2.5, 
                label='True Output', alpha=0.8, zorder=2)
        ax.plot(rollout_time, y_pred, color=colors[idx % len(colors)], 
                linestyle='--', linewidth=2, label='Predicted', alpha=0.8, zorder=3)
        
        ax.set_ylabel('Output y(t)', fontsize=11)
        ax.set_title(f'{model_name} - MSE: {mse_values[model_name]:.6f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if idx == len(predictions) - 1:
            ax.set_xlabel('Time Step', fontsize=11)
    
    plt.suptitle(f'{rollout_steps}-Step Rollout Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    
    return fig


def plot_training_history(history, model_name='Model', save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        model_name: Name of the model
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(best_epoch, best_val_loss, f'  Best: Epoch {best_epoch}', 
            fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    return fig

