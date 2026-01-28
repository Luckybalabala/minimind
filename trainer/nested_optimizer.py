"""
Nested Learning Optimizer

Implements multi-optimizer system for Nested Learning, where different
parameter groups update at different frequencies.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Any, Optional
import copy


class NestedOptimizer:
    """
    Nested optimizer that coordinates multiple optimizers.

    Each optimizer manages a parameter group with a specific update frequency.
    This enables multi-timescale learning where fast components adapt quickly
    while slow components change gradually.

    Attributes:
        model: The model being optimized
        config: Configuration object with nested learning settings
        optimizers: Dictionary mapping frequencies to optimizers
        param_groups: Dictionary mapping frequencies to parameter lists
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        base_optimizer_cls=optim.AdamW,
        base_lr: float = 1e-4
    ):
        """
        Initialize nested optimizer system.

        Args:
            model: PyTorch model to optimize
            config: MiniMindConfig with nested_learning settings
            base_optimizer_cls: Base optimizer class (default: AdamW)
            base_lr: Base learning rate
        """
        self.model = model
        self.config = config
        self.base_optimizer_cls = base_optimizer_cls
        self.base_lr = base_lr

        # Group parameters by update frequency
        self.param_groups = self._group_parameters_by_frequency()

        # Create optimizers for each frequency group
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        self.schedulers: Dict[int, Any] = {}

        if config.use_nested_learning and len(config.memory_frequencies) > 0:
            # Create optimizer for each frequency
            for freq, params in self.param_groups.items():
                if len(params) > 0:
                    # Scale learning rate by frequency (higher freq -> lower LR)
                    lr_scale = min(config.memory_frequencies) / freq
                    lr = base_lr * lr_scale

                    opt = base_optimizer_cls(
                        params,
                        lr=lr,
                        betas=(0.9, 0.999),
                        weight_decay=0.01
                    )
                    self.optimizers[freq] = opt

            # Create optimizer for slow/updating parameters (base model params)
            slow_params = self._get_slow_parameters()
            if len(slow_params) > 0:
                self.optimizers['slow'] = base_optimizer_cls(
                    slow_params,
                    lr=base_lr * 0.1,  # Lower LR for slow parameters
                    betas=(0.9, 0.999),
                    weight_decay=0.01
                )
        else:
            # Standard single optimizer mode
            all_params = list(model.parameters())
            self.optimizers['all'] = base_optimizer_cls(
                all_params,
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )

        # Track global step
        self.global_step = 0

    def _group_parameters_by_frequency(self) -> Dict[int, List[torch.nn.Parameter]]:
        """
        Group model parameters by their update frequency.

        Returns:
            Dictionary mapping frequencies to parameter lists
        """
        groups = {freq: [] for freq in self.config.memory_frequencies}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Group by name patterns
            if 'fast_memory' in name or 'memory_level_0' in name:
                # Fast memory parameters
                if len(self.config.memory_frequencies) > 0:
                    groups[self.config.memory_frequencies[0]].append(param)
            elif len(self.config.memory_frequencies) > 1 and ('memory_level_1' in name):
                # Medium frequency memory
                groups[self.config.memory_frequencies[1]].append(param)
            elif len(self.config.memory_frequencies) > 2 and ('memory_level_2' in name):
                # Slow memory
                groups[self.config.memory_frequencies[2]].append(param)
            # Other parameters will be in 'slow' group

        return groups

    def _get_slow_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get parameters that should update slowly (standard model params).

        Returns:
            List of parameters not in any frequency group
        """
        # Get all parameters that are in frequency groups
        freq_params = set()
        for params in self.param_groups.values():
            freq_params.update(id(p) for p in params)

        # Return parameters not in frequency groups
        slow_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in freq_params
        ]

        return slow_params

    def step(self, loss: Optional[torch.Tensor] = None):
        """
        Perform optimization step for appropriate optimizers.

        Args:
            loss: Optional loss tensor (for logging/analysis)
        """
        self.global_step += 1

        if not self.config.use_nested_learning:
            # Standard single optimizer mode
            if 'all' in self.optimizers:
                self.optimizers['all'].step()
                self.optimizers['all'].zero_grad()
            return

        # Always update slow optimizer
        if 'slow' in self.optimizers:
            self.optimizers['slow'].step()
            self.optimizers['slow'].zero_grad()

        # Update frequency-based optimizers according to schedule
        for freq, optimizer in self.optimizers.items():
            if freq == 'slow' or freq == 'all':
                continue

            # Check if this frequency should update
            if self.global_step % freq == 0:
                optimizer.step()
                optimizer.zero_grad()

    def zero_grad(self):
        """Zero gradients for all optimizers."""
        for opt in self.optimizers.values():
            opt.zero_grad()

    def get_lr(self) -> Dict[str, float]:
        """
        Get current learning rates for all optimizers.

        Returns:
            Dictionary mapping optimizer names to learning rates
        """
        lrs = {}
        for name, opt in self.optimizers.items():
            lrs[name] = [pg['lr'] for pg in opt.param_groups][0]
        return lrs

    def set_lr(self, lr: float, optimizer_name: str = 'slow'):
        """
        Set learning rate for a specific optimizer.

        Args:
            lr: New learning rate
            optimizer_name: Name of optimizer ('slow', or frequency number)
        """
        if optimizer_name in self.optimizers:
            for param_group in self.optimizers[optimizer_name].param_groups:
                param_group['lr'] = lr

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionaries for all optimizers.

        Returns:
            Nested dictionary with optimizer states
        """
        return {
            f'optimizer_{name}': opt.state_dict()
            for name, opt in self.optimizers.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state dictionaries for all optimizers.

        Args:
            state_dict: State dictionary from state_dict()
        """
        for name, state in state_dict.items():
            optimizer_key = name.replace('optimizer_', '')
            if optimizer_key in self.optimizers:
                self.optimizers[optimizer_key].load_state_dict(state)

    def get_update_stats(self) -> Dict[str, Any]:
        """
        Get statistics about parameter updates.

        Returns:
            Dictionary with update statistics
        """
        stats = {
            'global_step': self.global_step,
            'optimizers': {},
        }

        for name, opt in self.optimizers.items():
            # Count parameters in this optimizer
            num_params = sum(
                p.numel() for group in opt.param_groups for p in group['params']
            )

            stats['optimizers'][name] = {
                'num_parameters': num_params,
                'learning_rate': [pg['lr'] for pg in opt.param_groups][0],
                'updates': self.global_step if name in ['slow', 'all'] else self.global_step // int(name),
            }

        return stats

    def print_stats(self):
        """Print update statistics (useful for debugging)."""
        stats = self.get_update_stats()
        print(f"\n=== Nested Optimizer Stats (Step {stats['global_step']}) ===")
        for name, opt_stats in stats['optimizers'].items():
            print(f"{name}:")
            print(f"  Parameters: {opt_stats['num_parameters']:,}")
            print(f"  LR: {opt_stats['learning_rate']:.2e}")
            print(f"  Updates: {opt_stats['updates']}")
        print("=" * 50)


def create_nested_optimizer(
    model: torch.nn.Module,
    config: Any,
    learning_rate: float = 1e-4
) -> NestedOptimizer:
    """
    Factory function to create nested optimizer.

    Args:
        model: PyTorch model
        config: MiniMindConfig
        learning_rate: Base learning rate

    Returns:
        NestedOptimizer instance
    """
    return NestedOptimizer(
        model=model,
        config=config,
        base_optimizer_cls=optim.AdamW,
        base_lr=learning_rate
    )
