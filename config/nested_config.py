"""
Nested Learning Configuration Module

This module provides configuration classes for Nested Learning,
a multi-timescale learning paradigm that enables continual learning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class NestedLearningConfig:
    """
    Configuration for Nested Learning features.

    This class defines all parameters needed to enable and configure
    Nested Learning in MiniMind models.

    Attributes:
        enabled: Whether to enable Nested Learning (default: False for backward compatibility)
        num_memory_levels: Number of memory hierarchy levels (2 for MVP, 3 for full system)
        memory_frequencies: Update frequencies for each memory level
        memory_type: Type of memory update rule ('delta' or 'hebbian')
        fast_memory_dim: Dimension of fast memory (defaults to model hidden_size)
        enable_continual: Enable continual learning mode
        memory_replay_freq: Frequency of memory replay for continual learning
        optimizer_type: Type of optimizer to use ('nested_adamw', 'nested_sgd', etc.)
        learning_rates: Optional per-level learning rates
    """

    enabled: bool = False
    num_memory_levels: int = 2
    memory_frequencies: List[int] = field(default_factory=lambda: [64, 512])
    memory_type: str = 'delta'
    fast_memory_dim: Optional[int] = None
    enable_continual: bool = True
    memory_replay_freq: int = 1000

    # Optimizer configuration
    optimizer_type: str = 'nested_adamw'
    learning_rates: Optional[Dict[str, float]] = None

    # Memory parameters
    eta_g: float = 0.01  # Learning rate for fast memory
    alpha: float = 0.9   # Forgetting gate coefficient

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if self.learning_rates is None:
            # Default: scale learning rates by frequency
            self.learning_rates = {
                f"level_{i}": 1e-4 * (freq / max(self.memory_frequencies))
                for i, freq in enumerate(self.memory_frequencies)
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'num_memory_levels': self.num_memory_levels,
            'memory_frequencies': self.memory_frequencies,
            'memory_type': self.memory_type,
            'fast_memory_dim': self.fast_memory_dim,
            'enable_continual': self.enable_continual,
            'memory_replay_freq': self.memory_replay_freq,
            'optimizer_type': self.optimizer_type,
            'learning_rates': self.learning_rates,
            'eta_g': self.eta_g,
            'alpha': self.alpha,
        }

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NestedLearningConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'NestedLearningConfig':
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            AssertionError: If configuration is invalid
        """
        if not self.enabled:
            # If disabled, no need to validate further
            return True

        # Validate num_memory_levels
        assert self.num_memory_levels > 0, "num_memory_levels must be positive"

        # Validate memory_frequencies
        assert len(self.memory_frequencies) == self.num_memory_levels, \
            f"Number of frequencies ({len(self.memory_frequencies)}) must match " \
            f"num_memory_levels ({self.num_memory_levels})"

        assert all(f > 0 for f in self.memory_frequencies), \
            "All memory_frequencies must be positive"

        assert all(self.memory_frequencies[i] < self.memory_frequencies[i+1]
                   for i in range(len(self.memory_frequencies) - 1)), \
            "memory_frequencies must be in increasing order"

        # Validate memory_type
        assert self.memory_type in ['delta', 'hebbian'], \
            f"memory_type must be 'delta' or 'hebbian', got '{self.memory_type}'"

        # Validate learning rates
        if self.learning_rates is not None:
            assert len(self.learning_rates) == self.num_memory_levels, \
                f"Number of learning rates ({len(self.learning_rates)}) must match " \
                f"num_memory_levels ({self.num_memory_levels})"

            assert all(lr > 0 for lr in self.learning_rates.values()), \
                "All learning rates must be positive"

        # Validate memory parameters
        assert 0 < self.eta_g < 1, f"eta_g must be in (0, 1), got {self.eta_g}"
        assert 0 < self.alpha < 1, f"alpha must be in (0, 1), got {self.alpha}"

        return True

    def get_level_config(self, level: int) -> Dict[str, Any]:
        """
        Get configuration for a specific memory level.

        Args:
            level: Memory level index (0-based)

        Returns:
            Dictionary with level-specific configuration
        """
        assert 0 <= level < self.num_memory_levels, \
            f"level must be in [0, {self.num_memory_levels - 1}], got {level}"

        return {
            'level': level,
            'frequency': self.memory_frequencies[level],
            'learning_rate': self.learning_rates.get(f'level_{level}', 1e-4),
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"NestedLearningConfig(enabled={self.enabled}, levels={self.num_memory_levels})"


def create_default_config() -> NestedLearningConfig:
    """Create default Nested Learning configuration."""
    return NestedLearningConfig()


def create_mvp_config() -> NestedLearningConfig:
    """Create MVP configuration with 2 memory levels."""
    return NestedLearningConfig(
        enabled=True,
        num_memory_levels=2,
        memory_frequencies=[64, 512],
        memory_type='delta',
    )


def create_full_config() -> NestedLearningConfig:
    """Create full configuration with 3 memory levels."""
    return NestedLearningConfig(
        enabled=True,
        num_memory_levels=3,
        memory_frequencies=[64, 512, 4096],
        memory_type='delta',
        enable_continual=True,
    )
