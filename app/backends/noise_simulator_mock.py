import numpy as np
import torch


class NoiseSimulatorMock:
    """
    A mock quantum noise simulator that generates noisy quantum states for visualization.
    
    This simulator creates mock data for both single-qubit (Bloch sphere) and 
    two-qubit (Q-simplex) representations with configurable noise parameters.
    """
    
    def __init__(self, noise_level=0.1, seed=None):
        """
        Initialize the noise simulator.
        
        Args:
            noise_level (float): Amount of noise to add (0.0 to 1.0)
            seed (int, optional): Random seed for reproducibility
        """
        self.noise_level = noise_level
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
    def generate_single_qubit_states(self, num_states=10, pure=True):
        """
        Generate noisy single-qubit states.
        
        Args:
            num_states (int): Number of states to generate
            pure (bool): Whether to generate pure states (True) or mixed states (False)
            
        Returns:
            torch.Tensor: Complex tensor of shape (num_states, 2) representing qubit states
        """
        # Start with random pure states
        alpha = torch.rand(num_states) * torch.exp(1j * torch.rand(num_states) * 2 * np.pi)
        beta_mag = torch.sqrt(1 - torch.abs(alpha)**2)
        beta_phase = torch.rand(num_states) * 2 * np.pi
        beta = beta_mag * torch.exp(1j * beta_phase)
        
        states = torch.stack([alpha, beta], dim=1)
        
        if not pure:
            # Add noise to make mixed states
            noise = torch.randn(num_states, 2, dtype=torch.cfloat) * self.noise_level
            states = states + noise
            
            # Renormalize
            norms = torch.sqrt(torch.sum(torch.abs(states)**2, dim=1, keepdim=True))
            states = states / norms
            
        return states
    
    def generate_two_qubit_states(self, num_states=10, entangled_ratio=0.5):
        """
        Generate two-qubit states with varying degrees of entanglement.
        
        Args:
            num_states (int): Number of states to generate
            entangled_ratio (float): Ratio of entangled states to generate (0.0 to 1.0)
            
        Returns:
            torch.Tensor: Complex tensor of shape (num_states, 4) representing two-qubit states
        """
        states = torch.zeros((num_states, 4), dtype=torch.cfloat)
        
        # Determine how many entangled states to generate
        num_entangled = int(num_states * entangled_ratio)
        
        # Generate separable states
        for i in range(num_states - num_entangled):
            # Create separable state |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩
            q1_alpha = torch.rand(1) * torch.exp(1j * torch.rand(1) * 2 * np.pi)
            q1_beta = torch.sqrt(1 - torch.abs(q1_alpha)**2) * torch.exp(1j * torch.rand(1) * 2 * np.pi)
            
            q2_alpha = torch.rand(1) * torch.exp(1j * torch.rand(1) * 2 * np.pi)
            q2_beta = torch.sqrt(1 - torch.abs(q2_alpha)**2) * torch.exp(1j * torch.rand(1) * 2 * np.pi)
            
            # Tensor product
            states[i, 0] = q1_alpha * q2_alpha  # |00⟩
            states[i, 1] = q1_alpha * q2_beta   # |01⟩
            states[i, 2] = q1_beta * q2_alpha   # |10⟩
            states[i, 3] = q1_beta * q2_beta    # |11⟩
        
        # Generate entangled states (variations of Bell states with noise)
        for i in range(num_states - num_entangled, num_states):
            bell_type = np.random.randint(0, 4)
            
            if bell_type == 0:  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                states[i, 0] = 1/np.sqrt(2)
                states[i, 3] = 1/np.sqrt(2)
            elif bell_type == 1:  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                states[i, 0] = 1/np.sqrt(2)
                states[i, 3] = -1/np.sqrt(2)
            elif bell_type == 2:  # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                states[i, 1] = 1/np.sqrt(2)
                states[i, 2] = 1/np.sqrt(2)
            else:  # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
                states[i, 1] = 1/np.sqrt(2)
                states[i, 2] = -1/np.sqrt(2)
        
        # Add noise to all states
        if self.noise_level > 0:
            noise = torch.randn(num_states, 4, dtype=torch.cfloat) * self.noise_level
            states = states + noise
            
            # Renormalize
            norms = torch.sqrt(torch.sum(torch.abs(states)**2, dim=1, keepdim=True))
            states = states / norms
        
        return states
    
    def generate_mock_dataset(self, num_samples=100, num_classes=2):
        """
        Generate a mock dataset with labels for visualization.
        
        Args:
            num_samples (int): Number of samples to generate
            num_classes (int): Number of classes (2 or more)
            
        Returns:
            tuple: (single_qubit_states, two_qubit_states, labels)
                - single_qubit_states: torch.Tensor of shape (num_samples, 2)
                - two_qubit_states: torch.Tensor of shape (num_samples, 4)
                - labels: torch.Tensor of shape (num_samples,)
        """
        # Generate labels
        labels = torch.randint(0, num_classes, (num_samples,))
        
        # Generate states with some correlation to labels
        single_qubit_states = torch.zeros((num_samples, 2), dtype=torch.cfloat)
        two_qubit_states = torch.zeros((num_samples, 4), dtype=torch.cfloat)
        
        for c in range(num_classes):
            # Find indices for this class
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
                
            # Generate states for this class with some class-specific bias
            bias_angle = c * (2 * np.pi / num_classes)
            
            # Single qubit states biased toward a specific direction on Bloch sphere
            alpha = torch.cos(torch.tensor(bias_angle/2)) + 0j
            beta = torch.sin(torch.tensor(bias_angle/2)) * torch.exp(1j * torch.tensor(bias_angle))
            
            # Create template state for this class
            template_single = torch.tensor([alpha, beta], dtype=torch.cfloat)
            
            # Add noise to create variations within the class
            for i, index in enumerate(idx):
                noise = torch.randn(2, dtype=torch.cfloat) * self.noise_level
                state = template_single + noise
                # Renormalize
                norm = torch.sqrt(torch.sum(torch.abs(state)**2))
                single_qubit_states[index] = state / norm
            
            # Two qubit states - create class-specific entangled states
            if c % 2 == 0:  # Even classes get Φ-like states
                template_two = torch.zeros(4, dtype=torch.cfloat)
                template_two[0] = torch.cos(torch.tensor(bias_angle/2))
                template_two[3] = torch.sin(torch.tensor(bias_angle/2))
            else:  # Odd classes get Ψ-like states
                template_two = torch.zeros(4, dtype=torch.cfloat)
                template_two[1] = torch.cos(torch.tensor(bias_angle/2))
                template_two[2] = torch.sin(torch.tensor(bias_angle/2))
            
            # Add noise to create variations
            for i, index in enumerate(idx):
                noise = torch.randn(4, dtype=torch.cfloat) * self.noise_level
                state = template_two + noise
                # Renormalize
                norm = torch.sqrt(torch.sum(torch.abs(state)**2))
                two_qubit_states[index] = state / norm
        
        return single_qubit_states, two_qubit_states, labels


# Example usage
if __name__ == "__main__":
    simulator = NoiseSimulatorMock(noise_level=0.1, seed=42)
    
    # Generate individual states
    single_states = simulator.generate_single_qubit_states(num_states=5)
    two_states = simulator.generate_two_qubit_states(num_states=5)
    
    # Generate a dataset
    single_dataset, two_dataset, labels = simulator.generate_mock_dataset(
        num_samples=20, num_classes=3
    )
    
    print("Single qubit states shape:", single_states.shape)
    print("Two qubit states shape:", two_states.shape)
    print("Dataset shapes:", single_dataset.shape, two_dataset.shape, labels.shape)

