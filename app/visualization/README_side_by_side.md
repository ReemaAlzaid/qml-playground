# Side-by-Side Bloch Sphere/Q-Simplex Visualization Framework

This module provides a framework for visualizing quantum states in both the Bloch sphere (for single-qubit states) and the Q-simplex (for two-qubit states) side by side. This allows for a comprehensive view of quantum state evolution and entanglement properties.

## Features

- **Side-by-side visualization** of single-qubit and two-qubit states
- **Multi-layer visualization** for tracking state evolution through quantum circuit layers
- **Support for target states** to highlight specific quantum states
- **Class-based coloring** to distinguish different data classes
- **Customizable plot dimensions** and appearance

## Usage

### Basic Side-by-Side Visualization

```python
from app.backends.noise_simulator_mock import NoiseSimulatorMock
from app.visualization.side_by_side import make_side_by_side_visualization
import torch
import numpy as np

# Create mock data
simulator = NoiseSimulatorMock(noise_level=0.1)
single_states, two_states, labels = simulator.generate_mock_dataset(
    num_samples=50, num_classes=3
)

# Create target states (optional)
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

# Display the figure
fig.show()
```

### Multi-Layer Visualization

```python
from app.backends.noise_simulator_mock import NoiseSimulatorMock
from app.visualization.side_by_side import make_side_by_side_model_visualization
import torch
import numpy as np

# Create mock data
simulator = NoiseSimulatorMock(noise_level=0.1)
labels = torch.randint(0, 3, (50,))

# Generate sequences of states for multiple layers
num_layers = 3
single_seq = [simulator.generate_single_qubit_states(num_states=50) for _ in range(num_layers)]
two_seq = [simulator.generate_two_qubit_states(num_states=50) for _ in range(num_layers)]

# Create visualization
fig = make_side_by_side_model_visualization(
    single_qubit_states_seq=single_seq,
    two_qubit_states_seq=two_seq,
    labels=labels,
    num_layers=num_layers
)

# Display the figure
fig.show()
```

## Integration with QML Models

This visualization framework can be integrated with quantum machine learning models to visualize the evolution of quantum states during training:

```python
# Example with a QML model (pseudocode)
from app.models.reuploading_classifier import ReuploadingClassifier

# Initialize model
model = ReuploadingClassifier(...)

# Train model
model.train(...)

# Get state sequences from model
single_qubit_states = model.get_single_qubit_states()
two_qubit_states = model.get_two_qubit_states()

# Create visualization
fig = make_side_by_side_model_visualization(
    single_qubit_states_seq=single_qubit_states,
    two_qubit_states_seq=two_qubit_states,
    labels=labels
)

# Display the figure
fig.show()
```

## Noise Simulator Mock

The `NoiseSimulatorMock` class provides a way to generate mock quantum data with configurable noise levels and entanglement properties:

```python
from app.backends.noise_simulator_mock import NoiseSimulatorMock

# Create simulator with custom noise level
simulator = NoiseSimulatorMock(noise_level=0.2, seed=42)

# Generate single-qubit states
single_states = simulator.generate_single_qubit_states(num_states=100, pure=True)

# Generate two-qubit states with varying entanglement
two_states = simulator.generate_two_qubit_states(num_states=100, entangled_ratio=0.7)

# Generate a complete dataset with labels
single_dataset, two_dataset, labels = simulator.generate_mock_dataset(
    num_samples=200, num_classes=4
)
```

## Testing

You can run the test script to see examples of the visualizations:

```bash
python app/test_side_by_side.py
```

This will generate HTML files with different visualization examples:
- `side_by_side_single.html`: Basic side-by-side visualization
- `side_by_side_multi_layer.html`: Multi-layer visualization
- `side_by_side_evolution.html`: Visualization of state evolution with increasing entanglement