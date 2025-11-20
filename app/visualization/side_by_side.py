import logging

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from visualization.single_qubit import make_bloch_sphere, make_bloch_state_traces
from visualization.two_qubits import make_tetrahedron, make_tetrahedron_state_traces
from utils.qstate_representations import convert_state_vector_to_bloch_vector

logger = logging.getLogger(" [PLOTLY]")


def make_side_by_side_visualization(single_qubit_states=None, two_qubit_states=None, 
                                   labels=None, single_qubit_targets=None, 
                                   two_qubit_targets=None, title=None,
                                   width=800, height=400):
    """
    Create a side-by-side visualization of Bloch sphere and Q-simplex.
    
    Args:
        single_qubit_states: Tensor of single qubit states or Bloch vectors
        two_qubit_states: Tensor of two-qubit states
        labels: Class labels for the states
        single_qubit_targets: Target states for single qubit visualization
        two_qubit_targets: Target states for two-qubit visualization
        title: Title for the plot
        width: Width of the plot
        height: Height of the plot
        
    Returns:
        plotly.graph_objs.Figure: The side-by-side visualization figure
    """
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Bloch Sphere", "Q-Simplex"),
        horizontal_spacing=0.05
    )
    
    # Process single qubit states if provided
    if single_qubit_states is not None:
        # Check if we need to convert to Bloch vectors
        if single_qubit_states.shape[1] == 2:  # State vectors
            # Convert to Bloch vectors
            bloch_vectors = np.array([
                convert_state_vector_to_bloch_vector(state)
                for state in single_qubit_states.numpy()
            ])
        else:  # Already Bloch vectors
            bloch_vectors = single_qubit_states
    else:
        bloch_vectors = None
    
    # Convert target state vectors to Bloch vectors if needed
    bloch_targets = None
    if single_qubit_targets is not None:
        bloch_targets = np.array([
            convert_state_vector_to_bloch_vector(target)
            for target in single_qubit_targets.numpy()
        ])
    
    # Create Bloch sphere traces
    bloch_traces, bloch_annotations = make_bloch_sphere(targets=bloch_targets)
    bloch_state_traces = make_bloch_state_traces(bloch_vectors, labels)
    
    # Create Q-simplex (tetrahedron) traces
    tetra_traces, tetra_annotations = make_tetrahedron(targets=two_qubit_targets)
    tetra_state_traces = make_tetrahedron_state_traces(two_qubit_states, labels)
    
    # Add all traces to the figure
    for trace in bloch_traces:
        fig.add_trace(trace, row=1, col=1)
    
    for trace in bloch_state_traces:
        fig.add_trace(trace, row=1, col=1)
    
    for trace in tetra_traces:
        fig.add_trace(trace, row=1, col=2)
    
    for trace in tetra_state_traces:
        fig.add_trace(trace, row=1, col=2)
    
    # Configure the axes for both subplots
    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )
    
    # Camera positions
    bloch_camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.7, y=0.7, z=0.7)
    )
    
    tetra_camera = dict(
        eye=dict(x=1.4, y=1.1, z=0.2)
    )
    
    # Update layout for both subplots
    fig.update_layout(
        title=title,
        autosize=False,
        width=width,
        height=height,
        margin=dict(b=20, t=50, l=20, r=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update scene properties for Bloch sphere
    fig.update_scenes(
        row=1, col=1,
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        annotations=bloch_annotations,
        xaxis=axes_config,
        yaxis=axes_config,
        zaxis=axes_config,
        camera=bloch_camera
    )
    
    # Update scene properties for Q-simplex
    fig.update_scenes(
        row=1, col=2,
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        annotations=tetra_annotations,
        xaxis=axes_config,
        yaxis=axes_config,
        zaxis=axes_config,
        camera=tetra_camera,
        aspectratio=dict(x=1, y=1, z=1)
    )
    
    return fig


def make_side_by_side_model_visualization(single_qubit_states_seq=None, two_qubit_states_seq=None,
                                         labels=None, num_layers=None, single_qubit_targets=None,
                                         two_qubit_targets=None, width=1000, height=None):
    """
    Create a side-by-side visualization of Bloch sphere and Q-simplex for multiple layers.
    
    Args:
        single_qubit_states_seq: Sequence of single qubit states for each layer
        two_qubit_states_seq: Sequence of two-qubit states for each layer
        labels: Class labels for the states
        num_layers: Number of layers to visualize
        single_qubit_targets: Target states for single qubit visualization
        two_qubit_targets: Target states for two-qubit visualization
        width: Width of the plot
        height: Height of the plot (calculated automatically if None)
        
    Returns:
        plotly.graph_objs.Figure: The side-by-side visualization figure with multiple layers
    """
    if single_qubit_states_seq is None and two_qubit_states_seq is None and num_layers is None:
        raise ValueError("Must specify either state sequences or num_layers")
    
    # Determine number of layers
    if num_layers is None:
        if single_qubit_states_seq is not None:
            num_layers = len(single_qubit_states_seq)
        elif two_qubit_states_seq is not None:
            num_layers = len(two_qubit_states_seq)
    
    # Calculate height if not provided
    if height is None:
        height = 300 * num_layers
    
    # Create a subplot with num_layers rows and 2 columns
    fig = make_subplots(
        rows=num_layers, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}] for _ in range(num_layers)],
        subplot_titles=[f"Layer {i+1} - Bloch Sphere" for i in range(num_layers)] + 
                      [f"Layer {i+1} - Q-Simplex" for i in range(num_layers)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # Convert target state vectors to Bloch vectors if needed
    bloch_targets = None
    if single_qubit_targets is not None:
        bloch_targets = np.array([
            convert_state_vector_to_bloch_vector(target)
            for target in single_qubit_targets.numpy()
        ])
    
    # Create base traces for Bloch sphere and tetrahedron
    bloch_traces, bloch_annotations = make_bloch_sphere(targets=bloch_targets)
    tetra_traces, tetra_annotations = make_tetrahedron(targets=two_qubit_targets)
    
    # Add traces for each layer
    for layer in range(num_layers):
        # Add base traces to each subplot
        for trace in bloch_traces:
            fig.add_trace(trace, row=layer+1, col=1)
        
        for trace in tetra_traces:
            fig.add_trace(trace, row=layer+1, col=2)
        
        # Add state traces if available
        if single_qubit_states_seq is not None and layer < len(single_qubit_states_seq):
            single_states = single_qubit_states_seq[layer]
            
            # Check if we need to convert to Bloch vectors
            if single_states.shape[1] == 2:  # State vectors
                # Convert to Bloch vectors
                bloch_vectors = np.array([
                    convert_state_vector_to_bloch_vector(state) 
                    for state in single_states.numpy()
                ])
            else:  # Already Bloch vectors
                bloch_vectors = single_states
                
            bloch_state_traces = make_bloch_state_traces(bloch_vectors, labels)
            for trace in bloch_state_traces:
                fig.add_trace(trace, row=layer+1, col=1)
        
        if two_qubit_states_seq is not None and layer < len(two_qubit_states_seq):
            two_states = two_qubit_states_seq[layer]
            tetra_state_traces = make_tetrahedron_state_traces(two_states, labels)
            for trace in tetra_state_traces:
                fig.add_trace(trace, row=layer+1, col=2)
    
    # Configure the axes for all subplots
    axes_config = dict(
        backgroundcolor="white",
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
    )
    
    # Camera positions
    bloch_camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.7, y=0.7, z=0.7)
    )
    
    tetra_camera = dict(
        eye=dict(x=1.4, y=1.1, z=0.2)
    )
    
    # Update layout
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(b=20, t=50, l=20, r=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update scene properties for all subplots
    for layer in range(num_layers):
        # Bloch sphere
        fig.update_scenes(
            row=layer+1, col=1,
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            annotations=bloch_annotations,
            xaxis=axes_config,
            yaxis=axes_config,
            zaxis=axes_config,
            camera=bloch_camera
        )
        
        # Q-simplex
        fig.update_scenes(
            row=layer+1, col=2,
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            annotations=tetra_annotations,
            xaxis=axes_config,
            yaxis=axes_config,
            zaxis=axes_config,
            camera=tetra_camera,
            aspectratio=dict(x=1, y=1, z=1)
        )
    
    return fig


# Example usage
if __name__ == "__main__":
    import torch
    from backends.noise_simulator_mock import NoiseSimulatorMock
    
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
    
    # Create single visualization
    fig = make_side_by_side_visualization(
        single_qubit_states=single_states,
        two_qubit_states=two_states,
        labels=labels,
        single_qubit_targets=single_targets,
        two_qubit_targets=two_targets,
        title="Side-by-Side Bloch Sphere and Q-Simplex Visualization"
    )
    
    # Create multi-layer visualization
    # Generate sequences of states for multiple layers
    num_layers = 3
    single_seq = [simulator.generate_single_qubit_states(num_states=50) for _ in range(num_layers)]
    two_seq = [simulator.generate_two_qubit_states(num_states=50) for _ in range(num_layers)]
    
    multi_fig = make_side_by_side_model_visualization(
        single_qubit_states_seq=single_seq,
        two_qubit_states_seq=two_seq,
        labels=labels,
        single_qubit_targets=single_targets,
        two_qubit_targets=two_targets
    )
    
    # Save or display the figures
    # fig.show()
    # multi_fig.show()
