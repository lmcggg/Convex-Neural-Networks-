"""
Demonstration script for all network architectures on both test systems
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from networks import ICNN, ICRNN, CDiNN1, CDiNN2, RCDiNN
from data_generation import (
    generate_delay_system, 
    generate_nonlinear_narx,
    create_dataloaders
)
from train_utils import (
    train_rnn_model,
    train_feedforward_model,
    evaluate_rnn_model,
    evaluate_feedforward_model,
    compute_rollout_mse_rnn,
    verify_non_negative_weights
)
from visualize import (
    visualize_predictions,
    visualize_rollout_comparison,
    plot_training_history
)


def test_delay_system(device='cpu'):
    """
    Test on delay system: y_t = u_{t-4} + u_{t-3} + u_{t-2} + u_{t-1}
    This is a sanity check from the paper.
    """
    print("\n" + "="*70)
    print("TEST 1: DELAY SYSTEM (Sanity Check)")
    print("="*70)
    
    # Generate data
    u, y = generate_delay_system(n_samples=10000)
    
    # Create dataloaders
    window_size = 5
    train_loader, val_loader, test_loader, u_stats, y_stats = create_dataloaders(
        u, y, window_size=window_size, batch_size=32, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"\nData: {len(u)} samples, window_size={window_size}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Test R-CDiNN (should handle delay well)
    print("\n" + "-"*70)
    print("Training R-CDiNN on Delay System")
    print("-"*70)
    
    model_rcdinn = RCDiNN(input_dim=1, hidden_dim=32, output_dim=1, use_bias=False)
    model_rcdinn, history = train_rnn_model(
        model_rcdinn, train_loader, val_loader, 
        epochs=100, lr=1e-3, device=device, patience=15, verbose=True
    )
    
    test_mse = evaluate_rnn_model(model_rcdinn, test_loader, device=device)
    print(f"\nR-CDiNN Test MSE (1-step): {test_mse:.6f}")
    
    # Rollout test
    rollout_mse = compute_rollout_mse_rnn(
        model_rcdinn, u[-1000:], y[-1000:], window_size, rollout_steps=50,
        u_stats=u_stats, y_stats=y_stats, device=device
    )
    print(f"R-CDiNN Rollout MSE (50-step): {rollout_mse:.6f}")
    
    # Visualize R-CDiNN predictions
    print("\nGenerating R-CDiNN visualization...")
    visualize_predictions(
        model_rcdinn, u[-1000:], y[-1000:], window_size, 
        u_stats, y_stats, rollout_steps=50, device=device,
        model_name='R-CDiNN', save_path='results_rcdinn_delay.png'
    )
    
    # Plot training history
    plot_training_history(history, model_name='R-CDiNN (Delay System)', 
                         save_path='training_rcdinn_delay.png')
    
    # Test ICRNN (paper suggests it struggles with delay when bias is disabled)
    print("\n" + "-"*70)
    print("Training ICRNN on Delay System")
    print("-"*70)
    
    model_icrnn = ICRNN(input_dim=1, hidden_dim=32, output_dim=1, use_bias=False)
    model_icrnn, history = train_rnn_model(
        model_icrnn, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device, patience=15, verbose=True
    )
    
    test_mse = evaluate_rnn_model(model_icrnn, test_loader, device=device)
    print(f"\nICRNN Test MSE (1-step): {test_mse:.6f}")
    
    rollout_mse = compute_rollout_mse_rnn(
        model_icrnn, u[-1000:], y[-1000:], window_size, rollout_steps=50,
        u_stats=u_stats, y_stats=y_stats, device=device
    )
    print(f"ICRNN Rollout MSE (50-step): {rollout_mse:.6f}")
    
    # Visualize ICRNN predictions
    print("\nGenerating ICRNN visualization...")
    visualize_predictions(
        model_icrnn, u[-1000:], y[-1000:], window_size, 
        u_stats, y_stats, rollout_steps=50, device=device,
        model_name='ICRNN', save_path='results_icrnn_delay.png'
    )
    
    # Compare R-CDiNN vs ICRNN
    print("\nGenerating comparison visualization...")
    visualize_rollout_comparison(
        {'R-CDiNN': model_rcdinn, 'ICRNN': model_icrnn},
        u[-1000:], y[-1000:], window_size, u_stats, y_stats,
        rollout_steps=50, device=device, save_path='comparison_delay.png'
    )
    
    print("\n" + "="*70)


def test_nonlinear_narx(device='cpu'):
    """
    Test on nonlinear NARX system with saturation, cross terms, and delays
    This is the main evaluation system.
    """
    print("\n" + "="*70)
    print("TEST 2: NONLINEAR NARX SYSTEM (Main Evaluation)")
    print("="*70)
    
    # Generate data
    u, y = generate_nonlinear_narx(n_samples=50000, noise_std=0.01)
    
    # Create dataloaders
    window_size = 5
    train_loader, val_loader, test_loader, u_stats, y_stats = create_dataloaders(
        u, y, window_size=window_size, batch_size=64, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"\nData: {len(u)} samples, window_size={window_size}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    results = {}
    trained_models = {}  # Store models for comparison visualization
    
    # Test R-CDiNN
    print("\n" + "-"*70)
    print("Training R-CDiNN")
    print("-"*70)
    
    model_rcdinn = RCDiNN(input_dim=1, hidden_dim=64, output_dim=1, use_bias=False)
    model_rcdinn, history_rcdinn = train_rnn_model(
        model_rcdinn, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device, patience=20, verbose=True
    )
    
    test_mse = evaluate_rnn_model(model_rcdinn, test_loader, device=device)
    rollout_mse = compute_rollout_mse_rnn(
        model_rcdinn, u[-2000:], y[-2000:], window_size, rollout_steps=50,
        u_stats=u_stats, y_stats=y_stats, device=device
    )
    results['R-CDiNN'] = {'1-step': test_mse, '50-step': rollout_mse}
    trained_models['R-CDiNN'] = model_rcdinn
    print(f"\nR-CDiNN - Test MSE (1-step): {test_mse:.6f}, Rollout MSE (50-step): {rollout_mse:.6f}")
    verify_non_negative_weights(model_rcdinn)
    
    # Visualize R-CDiNN
    print("\nGenerating R-CDiNN visualization...")
    visualize_predictions(
        model_rcdinn, u[-2000:], y[-2000:], window_size, 
        u_stats, y_stats, rollout_steps=50, device=device,
        model_name='R-CDiNN', save_path='results_rcdinn_narx.png'
    )
    plot_training_history(history_rcdinn, model_name='R-CDiNN (NARX)', 
                         save_path='training_rcdinn_narx.png')
    
    # Test ICRNN
    print("\n" + "-"*70)
    print("Training ICRNN")
    print("-"*70)
    
    model_icrnn = ICRNN(input_dim=1, hidden_dim=64, output_dim=1, use_bias=False)
    model_icrnn, history_icrnn = train_rnn_model(
        model_icrnn, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device, patience=20, verbose=True
    )
    
    test_mse = evaluate_rnn_model(model_icrnn, test_loader, device=device)
    rollout_mse = compute_rollout_mse_rnn(
        model_icrnn, u[-2000:], y[-2000:], window_size, rollout_steps=50,
        u_stats=u_stats, y_stats=y_stats, device=device
    )
    results['ICRNN'] = {'1-step': test_mse, '50-step': rollout_mse}
    trained_models['ICRNN'] = model_icrnn
    print(f"\nICRNN - Test MSE (1-step): {test_mse:.6f}, Rollout MSE (50-step): {rollout_mse:.6f}")
    verify_non_negative_weights(model_icrnn)
    
    # Visualize ICRNN
    print("\nGenerating ICRNN visualization...")
    visualize_predictions(
        model_icrnn, u[-2000:], y[-2000:], window_size, 
        u_stats, y_stats, rollout_steps=50, device=device,
        model_name='ICRNN', save_path='results_icrnn_narx.png'
    )
    plot_training_history(history_icrnn, model_name='ICRNN (NARX)', 
                         save_path='training_icrnn_narx.png')
    
    # Test CDiNN-1 (feedforward)
    print("\n" + "-"*70)
    print("Training CDiNN-1 (Feedforward)")
    print("-"*70)
    
    model_cdinn1 = CDiNN1(input_dim=window_size, hidden_dims=[64, 64], output_dim=1, use_bias=False)
    model_cdinn1, _ = train_feedforward_model(
        model_cdinn1, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device, patience=20, verbose=True
    )
    
    test_mse = evaluate_feedforward_model(model_cdinn1, test_loader, device=device)
    results['CDiNN-1'] = {'1-step': test_mse}
    print(f"\nCDiNN-1 - Test MSE (1-step): {test_mse:.6f}")
    verify_non_negative_weights(model_cdinn1)
    
    # Test CDiNN-2 (feedforward)
    print("\n" + "-"*70)
    print("Training CDiNN-2 (Feedforward)")
    print("-"*70)
    
    model_cdinn2 = CDiNN2(input_dim=window_size, hidden_dims=[64, 64], output_dim=1, use_bias=False)
    model_cdinn2, _ = train_feedforward_model(
        model_cdinn2, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device, patience=20, verbose=True
    )
    
    test_mse = evaluate_feedforward_model(model_cdinn2, test_loader, device=device)
    results['CDiNN-2'] = {'1-step': test_mse}
    print(f"\nCDiNN-2 - Test MSE (1-step): {test_mse:.6f}")
    verify_non_negative_weights(model_cdinn2)
    
    # Test ICNN (feedforward baseline)
    print("\n" + "-"*70)
    print("Training ICNN (Feedforward)")
    print("-"*70)
    
    model_icnn = ICNN(input_dim=window_size, hidden_dims=[64, 64], use_bias=False, output_dim=1)
    model_icnn, _ = train_feedforward_model(
        model_icnn, train_loader, val_loader,
        epochs=100, lr=1e-3, device=device, patience=20, verbose=True
    )
    
    test_mse = evaluate_feedforward_model(model_icnn, test_loader, device=device)
    results['ICNN'] = {'1-step': test_mse}
    print(f"\nICNN - Test MSE (1-step): {test_mse:.6f}")
    verify_non_negative_weights(model_icnn)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY - NONLINEAR NARX SYSTEM")
    print("="*70)
    print(f"{'Model':<15} {'1-step MSE':<15} {'50-step Rollout MSE':<20}")
    print("-"*70)
    for model_name, metrics in results.items():
        one_step = f"{metrics['1-step']:.6f}"
        rollout = f"{metrics.get('50-step', 'N/A')}" if isinstance(metrics.get('50-step', 'N/A'), str) else f"{metrics['50-step']:.6f}"
        print(f"{model_name:<15} {one_step:<15} {rollout:<20}")
    print("="*70)
    
    # Generate comparison visualization for RNN models
    print("\nGenerating final comparison visualization for RNN models...")
    visualize_rollout_comparison(
        trained_models,
        u[-2000:], y[-2000:], window_size, u_stats, y_stats,
        rollout_steps=50, device=device, save_path='comparison_narx.png'
    )
    
    print("\nAll visualizations saved successfully!")
    print("  - Individual model results: results_*_narx.png")
    print("  - Training histories: training_*_narx.png")
    print("  - Model comparison: comparison_narx.png")


def main():
    """Main demonstration"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run tests
    test_delay_system(device=device)
    test_nonlinear_narx(device=device)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()

