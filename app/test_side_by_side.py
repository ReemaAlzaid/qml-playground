import numpy as np
import torch
import plotly.io as pio

from app.backends.noise_simulator_mock import NoiseSimulatorMock
from app.visualization.side_by_side import (
    make_side_by_side_visualization,
    make_side_by_side_model_visualization
)


def test_single_visualization():
    """Test the side-by-side visualization with a single view."""
    print("Testing single side-by-side visualization...")
    
    # Create mock data
    simulator = NoiseSimulatorMock(noise_level=0.1, seed=42)
    single_states, two_states, labels = simulator.generate_mock_dataset(
        num_samples=50, num_classes=3
    )
    
    # Create target states
    single_targets = torch.tensor([
        [1.0, 0.0],  # |0⟩
        [0.0, 1.0],  # |1⟩
        [1/np.sqrt(2), 1/np.sqrt(2)],  # |+⟩
        [1/np.sqrt(2), -1/np.sqrt(2)]  # |-⟩
    ])
    
    two_targets = (
        [0, 1, 2, 3],  # Target indices
        [0, 1, 2, 3]   # Target states
    )
    
    # Create visualization
    fig = make_side_by_side_visualization(
        single_qubit_states=single_states,
        two_qubit_states=two_states,
        labels=labels,
        single_qubit_targets=single_targets,
        two_qubit_targets=two_targets,
        title="Side-by-Side Bloch Sphere and Q-Simplex Visualization"
    )
    
    # Save the figure
    pio.write_html(fig, "side_by_side_single.html")
    print("Saved visualization to side_by_side_single.html")
    
    return fig


def test_multi_layer_visualization():
    """Test the side-by-side visualization with multiple layers."""
    print("Testing multi-layer side-by-side visualization...")
    
    # Create mock data
    simulator = NoiseSimulatorMock(noise_level=0.1, seed=42)
    labels = torch.randint(0, 3, (50,))
    
    # Create target states
    single_targets = torch.tensor([
        [1.0, 0.0],  # |0⟩
        [0.0, 1.0],  # |1⟩
        [1/np.sqrt(2), 1/np.sqrt(2)],  # |+⟩
        [1/np.sqrt(2), -1/np.sqrt(2)]  # |-⟩
    ])
    
    two_targets = (
        [0, 1, 2, 3],  # Target indices
        [0, 1, 2, 3]   # Target states
    )
    
    # Generate sequences of states for multiple layers
    num_layers = 3
    single_seq = [simulator.generate_single_qubit_states(num_states=50) for _ in range(num_layers)]
    two_seq = [simulator.generate_two_qubit_states(num_states=50) for _ in range(num_layers)]
    
    # Create visualization
    fig = make_side_by_side_model_visualization(
        single_qubit_states_seq=single_seq,
        two_qubit_states_seq=two_seq,
        labels=labels,
        single_qubit_targets=single_targets,
        two_qubit_targets=two_targets
    )
    
    # Save the figure
    pio.write_html(fig, "side_by_side_multi_layer.html")
    print("Saved visualization to side_by_side_multi_layer.html")
    
    return fig


def test_evolution_visualization():
    """Test visualization of state evolution with increasing entanglement."""
    print("Testing state evolution visualization...")
    
    # Create mock data with increasing entanglement
    num_samples = 50
    num_layers = 4
    labels = torch.randint(0, 3, (num_samples,))
    
    # Create sequences with increasing entanglement
    single_seq = []
    two_seq = []
    
    for layer in range(num_layers):
        # Increase noise and entanglement with each layer
        noise_level = 0.05 * (layer + 1)
        entangled_ratio = 0.2 * (layer + 1)
        
        simulator = NoiseSimulatorMock(noise_level=noise_level, seed=42+layer)
        
        # Generate states
        single_states = simulator.generate_single_qubit_states(num_states=num_samples)
        two_states = simulator.generate_two_qubit_states(
            num_states=num_samples, 
            entangled_ratio=min(entangled_ratio, 1.0)
        )
        
        single_seq.append(single_states)
        two_seq.append(two_states)
    
    # Create target states
    single_targets = torch.tensor([
        [1.0, 0.0],  # |0⟩
        [0.0, 1.0],  # |1⟩
        [1/np.sqrt(2), 1/np.sqrt(2)],  # |+⟩
        [1/np.sqrt(2), -1/np.sqrt(2)]  # |-⟩
    ])
    
    two_targets = (
        [0, 1, 2, 3],  # Target indices
        [0, 1, 2, 3]   # Target states
    )
    
    # Create visualization
    fig = make_side_by_side_model_visualization(
        single_qubit_states_seq=single_seq,
        two_qubit_states_seq=two_seq,
        labels=labels,
        single_qubit_targets=single_targets,
        two_qubit_targets=two_targets
    )
    
    # Save the figure
    pio.write_html(fig, "side_by_side_evolution.html")
    print("Saved visualization to side_by_side_evolution.html")
    
    return fig


if __name__ == "__main__":
    print("Running side-by-side visualization tests...")
    
    # Run all tests
    test_single_visualization()
    test_multi_layer_visualization()
    test_evolution_visualization()
    
    print("All tests completed. Check the HTML files for visualizations.")

