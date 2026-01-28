"""
Nested Learning Memory Modules

This module implements memory components for Nested Learning, including
fast-adapting memory using Delta Rule and Hebbian learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FastMemoryModule(nn.Module):
    """
    Fast-adapting memory module using Delta Rule.

    This module implements a Level 1 memory component that updates online
    using the Delta Rule: M ← αM + η_g * (error * x^T)

    The memory matrix M is updated during forward pass to enable rapid
    adaptation to new patterns.

    Attributes:
        dim: Dimension of the memory matrix
        memory: Learnable memory matrix (dim x dim)
        eta_g: Learning rate parameter
        alpha: Forgetting gate coefficient
        reset: Optional reset gate for delta updates
    """

    def __init__(self, config):
        """
        Initialize FastMemoryModule.

        Args:
            config: MiniMindConfig or dict with configuration parameters
        """
        super().__init__()

        # Extract configuration
        if hasattr(config, 'fast_memory_dim'):
            self.dim = config.fast_memory_dim
        else:
            self.dim = config.hidden_size if hasattr(config, 'hidden_size') else 512

        if hasattr(config, 'memory_type'):
            self.memory_type = config.memory_type
        else:
            self.memory_type = 'delta'

        # Memory matrix - initialized to zeros
        self.memory = nn.Parameter(torch.zeros(self.dim, self.dim))

        # Learnable parameters
        self.eta_g = nn.Parameter(torch.tensor(0.01))  # Learning rate
        self.alpha = nn.Parameter(torch.tensor(0.9))   # Forgetting gate

        # Optional reset gate for delta rule
        if self.memory_type == 'delta':
            self.reset = nn.Linear(self.dim, self.dim, bias=False)
            nn.init.zeros_(self.reset.weight)

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Forward pass with optional online update.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            update: Whether to update memory (default: True)

        Returns:
            Retrieved tensor of shape (batch, seq_len, dim)
        """
        bsz, seq_len, dim = x.shape

        # Retrieve from memory: M * x
        # Expand memory for batch processing
        memory_expanded = self.memory.unsqueeze(0).expand(bsz, -1, -1)
        retrieved = torch.bmm(x, memory_expanded)

        # Online update (only during training or when continual learning is enabled)
        if update and (self.training or hasattr(self, 'enable_continual')):
            # Clamp learning rate to prevent instability
            eta_g_clamped = torch.clamp(self.eta_g, 0.0, 0.1)
            alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)

            # Compute error signal
            error = x - retrieved

            # Delta rule update: ΔM = η_g * (error * x^T) / seq_len
            delta = torch.bmm(error.transpose(1, 2), x) / seq_len
            delta_mean = delta.mean(dim=0)  # Average over batch

            # Apply reset gate if using delta rule
            if self.memory_type == 'delta' and hasattr(self, 'reset'):
                # Compute global context: average over batch and sequence
                context = x.mean(dim=(0, 1))  # (dim,)
                reset_effect = self.reset(context)  # (dim,)
                delta_mean = delta_mean + reset_effect.unsqueeze(1)  # (dim, 1) -> (dim, dim) broadcast

            # Update memory with forgetting gate
            with torch.no_grad():
                self.memory.data = alpha_clamped * self.memory.data + eta_g_clamped * delta_mean

        return retrieved

    def reset_memory(self):
        """Reset memory matrix to zeros."""
        with torch.no_grad():
            self.memory.zero_()

    def get_memory_norm(self) -> float:
        """Get the current norm of the memory matrix."""
        return self.memory.norm().item()


class HebbianMemoryModule(nn.Module):
    """
    Hebbian learning memory module.

    Implements Hebbian learning rule: ΔM = η * (x * x^T)
    Strengthens connections based on correlation.

    Attributes:
        dim: Dimension of the memory matrix
        memory: Learnable memory matrix
        eta_h: Hebbian learning rate
        alpha: Decay coefficient
    """

    def __init__(self, config):
        """
        Initialize HebbianMemoryModule.

        Args:
            config: MiniMindConfig with configuration parameters
        """
        super().__init__()

        if hasattr(config, 'fast_memory_dim'):
            self.dim = config.fast_memory_dim
        else:
            self.dim = config.hidden_size if hasattr(config, 'hidden_size') else 512

        # Memory matrix
        self.memory = nn.Parameter(torch.zeros(self.dim, self.dim))

        # Learnable parameters
        self.eta_h = nn.Parameter(torch.tensor(0.01))  # Hebbian learning rate
        self.alpha = nn.Parameter(torch.tensor(0.95))  # Decay

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Forward pass with Hebbian update.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            update: Whether to update memory

        Returns:
            Retrieved tensor of shape (batch, seq_len, dim)
        """
        bsz, seq_len, dim = x.shape

        # Retrieve
        memory_expanded = self.memory.unsqueeze(0).expand(bsz, -1, -1)
        retrieved = torch.bmm(x, memory_expanded)

        # Hebbian update
        if update:
            eta_h_clamped = torch.clamp(self.eta_h, 0.0, 0.1)
            alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)

            # Hebbian rule: ΔM = η * (x * x^T) / seq_len
            hebbian_update = torch.bmm(x.transpose(1, 2), x) / seq_len
            hebbian_mean = hebbian_update.mean(dim=0)

            # Apply update with decay
            with torch.no_grad():
                self.memory.data = alpha_clamped * self.memory.data + eta_h_clamped * hebbian_mean

        return retrieved

    def reset_memory(self):
        """Reset memory matrix to zeros."""
        with torch.no_grad():
            self.memory.zero_()


class SlowMemoryModule(nn.Module):
    """
    Slow-updating memory module for long-term knowledge.

    This module stores stable, long-term patterns and updates
    at a much lower frequency than fast memory.

    Attributes:
        dim: Dimension of the memory
        memory: Learnable memory matrix
        update_counter: Counter for tracking updates
    """

    def __init__(self, config):
        """
        Initialize SlowMemoryModule.

        Args:
            config: MiniMindConfig with configuration parameters
        """
        super().__init__()

        if hasattr(config, 'fast_memory_dim'):
            self.dim = config.fast_memory_dim
        else:
            self.dim = config.hidden_size if hasattr(config, 'hidden_size') else 512

        # Use standard linear layer for slow memory
        self.memory = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.xavier_uniform_(self.memory.weight)

        # Update tracking
        self.register_buffer('update_counter', torch.tensor(0))

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Forward pass through slow memory.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            update: Whether this is an update step (for tracking)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Apply standard linear transformation
        output = self.memory(x)

        # Track updates
        if update and self.training:
            self.update_counter += 1

        return output

    def get_update_count(self) -> int:
        """Get the number of updates performed."""
        return self.update_counter.item()


def create_memory_module(config, level: int = 0):
    """
    Factory function to create memory modules based on level.

    Args:
        config: MiniMindConfig
        level: Memory level (0=fast, 1=medium, 2=slow)

    Returns:
        Appropriate memory module
    """
    if level == 0:
        # Fast memory
        if hasattr(config, 'memory_type') and config.memory_type == 'hebbian':
            return HebbianMemoryModule(config)
        else:
            return FastMemoryModule(config)
    elif level == 1:
        # Medium memory - could be FFN or another fast memory
        return FastMemoryModule(config)
    else:
        # Slow memory
        return SlowMemoryModule(config)
