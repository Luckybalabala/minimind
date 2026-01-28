"""
Test-Time Scaling for MiniMind with Nested Learning

Implements test-time adaptation mechanisms that allow the model to adapt
to individual inputs or small batches during inference without full training.

Based on:
- Test-Time Training (TTT)
- Self-supervised test-time adaptation
- Gradient-based adaptation with momentum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import copy


@dataclass
class AdaptationConfig:
    """Configuration for test-time adaptation"""
    # Adaptation strategy
    strategy: str = "hybrid"  # "gradient", "memory", "hybrid"

    # Gradient-based adaptation
    num_steps: int = 5
    adaptation_lr: float = 1e-4
    momentum: float = 0.9
    gradient_clip: float = 1.0

    # Memory-based adaptation
    use_fast_memory: bool = True
    fast_memory_lr: float = 1e-3

    # Safety mechanisms
    max_parameter_drift: float = 0.1  # Max relative change from original params
    use_ema: bool = True  # Exponential moving average for recovery
    ema_decay: float = 0.999

    # Adaptation objectives
    use_self_supervised: bool = True
    use_consistency: bool = True
    use_entropy_minimization: bool = True

    # Regularization
    regularization_weight: float = 0.01
    consistency_weight: float = 0.1
    entropy_weight: float = 0.05


class TestTimeAdapter(nn.Module):
    """
    Test-Time Adapter for model adaptation during inference

    Supports multiple adaptation strategies:
    1. Gradient-based: Fine-tune on current input using self-supervised objectives
    2. Memory-based: Use fast memory for quick adaptation
    3. Hybrid: Combine both approaches
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptationConfig,
        device: str = "cuda"
    ):
        super().__init__()
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Store original parameters for recovery
        self.original_state = None
        self.adapted_state = None

        # EMA parameters for recovery
        self.ema_state = None

        # Adaptation history
        self.adaptation_history = {
            'losses': [],
            'parameter_drift': [],
            'adaptation_steps': []
        }

        # Adaptation parameters (subset of model parameters)
        self.adaptation_params = self._select_adaptation_parameters()

    def _select_adaptation_parameters(self) -> List[nn.Parameter]:
        """
        Select which parameters to adapt during test-time

        Strategy: Adapt faster-changing parameters (recent layers, attention)
        """
        adapt_params = []

        for name, param in self.model.named_parameters():
            # Adapt: attention layers, normalization, recent layers
            if any(key in name.lower() for key in ['attn', 'norm', 'layer', 'memory']):
                if param.requires_grad:
                    adapt_params.append(param)

        return adapt_params

    def save_original_state(self):
        """Save original model state before adaptation"""
        self.original_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Initialize EMA with original state
        if self.config.use_ema:
            self.ema_state = {
                name: param.clone().detach()
                for name, param in self.original_state.items()
            }

    def restore_original_state(self):
        """Restore original model parameters"""
        if self.original_state is not None:
            for name, param in self.model.named_parameters():
                if name in self.original_state:
                    param.data = self.original_state[name].data.clone()

    def restore_ema_state(self):
        """Restore EMA state"""
        if self.ema_state is not None:
            for name, param in self.model.named_parameters():
                if name in self.ema_state:
                    param.data = self.ema_state[name].data.clone()

    def update_ema(self):
        """Update EMA with current parameters"""
        if self.ema_state is not None:
            for name, param in self.model.named_parameters():
                if name in self.ema_state:
                    self.ema_state[name].data = (
                        self.config.ema_decay * self.ema_state[name].data +
                        (1 - self.config.ema_decay) * param.data
                    )

    def compute_parameter_drift(self) -> float:
        """
        Compute average parameter drift from original state

        Returns:
            Average relative parameter change
        """
        if self.original_state is None:
            return 0.0

        total_drift = 0.0
        num_params = 0

        for name, param in self.model.named_parameters():
            if name in self.original_state:
                original = self.original_state[name]
                drift = torch.norm(param.data - original.data) / (torch.norm(original.data) + 1e-8)
                total_drift += drift.item()
                num_params += 1

        return total_drift / num_params if num_params > 0 else 0.0

    def check_safety_constraints(self) -> bool:
        """
        Check if adaptation is safe (within parameter drift limits)

        Returns:
            True if safe, False if drift exceeds limits
        """
        drift = self.compute_parameter_drift()
        return drift <= self.config.max_parameter_drift

    def adapt(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Adapt model to current input using test-time training

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for supervised adaptation
            num_steps: Override default num_steps

        Returns:
            Dictionary with adaptation results
        """
        if self.original_state is None:
            self.save_original_state()

        num_steps = num_steps or self.config.num_steps
        losses = []

        # Set model to training mode for adaptation
        self.model.train()

        for step in range(num_steps):
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Compute adaptation loss
            loss = self._compute_adaptation_loss(
                outputs, input_ids, attention_mask, labels
            )

            losses.append(loss.item())

            # Backward and optimize
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.adaptation_params,
                self.config.gradient_clip
            )

            # Update parameters
            with torch.no_grad():
                for param in self.adaptation_params:
                    if param.grad is not None:
                        # SGD with momentum
                        if not hasattr(param, 'velocity'):
                            param.velocity = torch.zeros_like(param.data)

                        param.velocity = (
                            self.config.momentum * param.velocity -
                            self.config.adaptation_lr * param.grad
                        )
                        param.data += param.velocity

            # Zero gradients
            self.model.zero_grad()

            # Update EMA
            if self.config.use_ema:
                self.update_ema()

            # Check safety
            if not self.check_safety_constraints():
                print(f"⚠️  Safety constraint violated at step {step}, restoring EMA state")
                self.restore_ema_state()
                break

        # Record history
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        final_drift = self.compute_parameter_drift()

        self.adaptation_history['losses'].append(avg_loss)
        self.adaptation_history['parameter_drift'].append(final_drift)
        self.adaptation_history['adaptation_steps'].append(len(losses))

        return {
            'avg_loss': avg_loss,
            'final_drift': final_drift,
            'num_steps': len(losses),
            'losses': losses
        }

    def _compute_adaptation_loss(
        self,
        outputs: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total adaptation loss with multiple objectives

        Objectives:
        1. Supervised loss (if labels provided)
        2. Self-supervised loss (masked language modeling)
        3. Consistency regularization
        4. Entropy minimization
        """
        total_loss = 0.0
        loss_components = {}

        # Supervised loss (if labels available)
        if labels is not None and hasattr(outputs, 'loss') and outputs.loss is not None:
            supervised_loss = outputs.loss
            total_loss += supervised_loss
            loss_components['supervised'] = supervised_loss.item()

        # Self-supervised loss: masked language modeling
        if self.config.use_self_supervised:
            # Create random mask
            mask = torch.rand_like(input_ids.float()) < 0.15
            masked_input = input_ids.clone()
            # Use 0 as mask token if no mask_token_id available
            mask_token_id = 0

            # Forward with masked input
            with torch.no_grad():
                target_outputs = self.model(input_ids, attention_mask=attention_mask)

            masked_outputs = self.model(masked_input, attention_mask=attention_mask)

            # Compute KL divergence between original and masked
            if hasattr(target_outputs, 'logits') and hasattr(masked_outputs, 'logits'):
                kl_loss = F.kl_div(
                    F.log_softmax(masked_outputs.logits, dim=-1),
                    F.softmax(target_outputs.logits, dim=-1),
                    reduction='batchmean'
                )
                total_loss += self.config.regularization_weight * kl_loss
                loss_components['self_supervised'] = kl_loss.item()

        # Consistency regularization: output should be consistent under noise
        if self.config.use_consistency and attention_mask is not None:
            # Add noise to attention mask
            noisy_mask = attention_mask * torch.rand_like(attention_mask.float()).uniform_(0.9, 1.0)

            noisy_outputs = self.model(input_ids, attention_mask=noisy_mask)

            if hasattr(outputs, 'logits') and hasattr(noisy_outputs, 'logits'):
                consistency_loss = F.mse_loss(
                    outputs.logits, noisy_outputs.logits
                )
                total_loss += self.config.consistency_weight * consistency_loss
                loss_components['consistency'] = consistency_loss.item()

        # Entropy minimization: confident predictions
        if self.config.use_entropy_minimization and hasattr(outputs, 'logits'):
            # Compute entropy
            probs = F.softmax(outputs.logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            # Minimize entropy
            total_loss += self.config.entropy_weight * entropy
            loss_components['entropy'] = entropy.item()

        return total_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adapt: bool = True
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Forward pass with optional test-time adaptation

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels
            adapt: Whether to perform test-time adaptation

        Returns:
            Tuple of (model outputs, adaptation_info)
        """
        adaptation_info = None

        if adapt:
            # Adapt model to input
            adaptation_info = self.adapt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Set to eval mode for inference
        self.model.eval()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        return outputs, adaptation_info

    def reset(self):
        """Reset to original state and clear history"""
        self.restore_original_state()
        self.adaptation_history = {
            'losses': [],
            'parameter_drift': [],
            'adaptation_steps': []
        }

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        if not self.adaptation_history['losses']:
            return {'status': 'No adaptation performed'}

        return {
            'num_adaptations': len(self.adaptation_history['losses']),
            'avg_loss': sum(self.adaptation_history['losses']) / len(self.adaptation_history['losses']),
            'avg_drift': sum(self.adaptation_history['parameter_drift']) / len(self.adaptation_history['parameter_drift']),
            'avg_steps': sum(self.adaptation_history['adaptation_steps']) / len(self.adaptation_history['adaptation_steps']),
            'total_steps': sum(self.adaptation_history['adaptation_steps'])
        }


class MemoryBasedAdapter(nn.Module):
    """
    Memory-based test-time adapter using fast memory

    Leverages the fast memory module from Nested Learning for rapid adaptation
    without modifying the main model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        fast_memory_lr: float = 1e-3,
        device: str = "cuda"
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.fast_memory_lr = fast_memory_lr

        # Extract fast memory modules from model
        self.fast_memory_modules = self._find_fast_memory_modules()

        # Original memory states
        self.original_memory_states = {}

    def _find_fast_memory_modules(self) -> List[nn.Module]:
        """Find FastMemoryModule instances in the model"""
        memory_modules = []
        
        # Try to import, handle if not available
        try:
            from model.nested_memory import FastMemoryModule
            for module in self.model.modules():
                if isinstance(module, FastMemoryModule):
                    memory_modules.append(module)
        except ImportError:
            pass
            
        return memory_modules

    def save_memory_states(self):
        """Save original fast memory states"""
        for i, module in enumerate(self.fast_memory_modules):
            if hasattr(module, 'memory'):
                self.original_memory_states[i] = module.memory.data.clone()

    def restore_memory_states(self):
        """Restore original fast memory states"""
        for i, module in enumerate(self.fast_memory_modules):
            if i in self.original_memory_states and hasattr(module, 'memory'):
                module.memory.data = self.original_memory_states[i].clone()

    def adapt_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Adapt fast memory to current input

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels
            num_steps: Number of adaptation steps

        Returns:
            Adaptation info dictionary
        """
        if not self.fast_memory_modules:
            return {'status': 'No fast memory modules found'}

        if not self.original_memory_states:
            self.save_memory_states()

        losses = []

        for step in range(num_steps):
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss_val = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0)
            losses.append(loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val)

            # Backward pass (only for fast memory)
            if isinstance(loss_val, torch.Tensor) and loss_val != 0.0:
                loss_val.backward()

                # Update only fast memory parameters
                with torch.no_grad():
                    for module in self.fast_memory_modules:
                        if hasattr(module, 'memory') and module.memory.grad is not None:
                            module.memory.data -= self.fast_memory_lr * module.memory.grad.data
                            module.memory.grad.zero_()

        return {
            'avg_loss': sum(losses) / len(losses) if losses else 0.0,
            'num_steps': num_steps,
            'num_modules': len(self.fast_memory_modules)
        }

    def reset(self):
        """Reset fast memory to original state"""
        self.restore_memory_states()


class HybridAdapter(nn.Module):
    """
    Hybrid adapter combining gradient-based and memory-based adaptation

    Uses both parameter adaptation and fast memory for maximum flexibility.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptationConfig,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        # Create both adapters
        self.gradient_adapter = TestTimeAdapter(model, config, device)
        self.memory_adapter = MemoryBasedAdapter(
            model,
            config.fast_memory_lr,
            device
        )
        self.config = config

    def adapt(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Hybrid adaptation: combine gradient and memory-based approaches

        Strategy: Alternate between gradient and memory updates
        """
        num_steps = num_steps or self.config.num_steps

        gradient_steps = num_steps // 2
        memory_steps = num_steps - gradient_steps

        results = {}

        # Gradient-based adaptation
        if gradient_steps > 0:
            gradient_results = self.gradient_adapter.adapt(
                input_ids, attention_mask, labels, gradient_steps
            )
            results['gradient'] = gradient_results

        # Memory-based adaptation
        if memory_steps > 0:
            memory_results = self.memory_adapter.adapt_memory(
                input_ids, attention_mask, labels, memory_steps
            )
            results['memory'] = memory_results

        # Combined stats
        results['combined'] = {
            'total_steps': num_steps,
            'gradient_steps': gradient_steps,
            'memory_steps': memory_steps
        }

        return results

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adapt: bool = True
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Forward pass with optional hybrid adaptation"""
        adaptation_info = None

        if adapt:
            adaptation_info = self.adapt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        self.gradient_adapter.model.eval()

        with torch.no_grad():
            outputs = self.gradient_adapter.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        return outputs, adaptation_info

    def reset(self):
        """Reset both adapters"""
        self.gradient_adapter.reset()
        self.memory_adapter.reset()

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get combined adaptation statistics"""
        return {
            'gradient': self.gradient_adapter.get_adaptation_stats(),
            'memory': {
                'num_modules': len(self.memory_adapter.fast_memory_modules)
            }
        }


def create_test_time_adapter(
    model: nn.Module,
    strategy: str = "hybrid",
    config: Optional[AdaptationConfig] = None,
    device: str = "cuda"
) -> nn.Module:
    """
    Factory function to create test-time adapter

    Args:
        model: PyTorch model to adapt
        strategy: Adaptation strategy ("gradient", "memory", "hybrid")
        config: Adaptation configuration
        device: Device to use

    Returns:
        Test-time adapter instance
    """
    if config is None:
        config = AdaptationConfig()

    config.strategy = strategy

    if strategy == "gradient":
        return TestTimeAdapter(model, config, device)
    elif strategy == "memory":
        return MemoryBasedAdapter(model, config.fast_memory_lr, device)
    elif strategy == "hybrid":
        return HybridAdapter(model, config, device)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class TestTimeBatchAdapter:
    """
    Batch-level test-time adaptation for processing multiple samples efficiently

    Adapts model to a batch of inputs, handling diverse samples within the batch.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptationConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Create adapter
        self.adapter = create_test_time_adapter(model, config.strategy, config, device)

    def adapt_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adapt_per_sample: bool = False
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Adapt to batch of inputs

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels [batch_size, seq_len]
            adapt_per_sample: If True, adapt to each sample separately

        Returns:
            Tuple of (model outputs, list of adaptation info per sample)
        """
        batch_size = input_ids.size(0)
        adaptation_infos = []

        if adapt_per_sample:
            # Adapt to each sample separately
            outputs_list = []

            for i in range(batch_size):
                # Reset adapter
                self.adapter.reset()

                # Adapt to single sample
                sample_input = input_ids[i:i+1]
                sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
                sample_labels = labels[i:i+1] if labels is not None else None

                outputs, info = self.adapter(
                    input_ids=sample_input,
                    attention_mask=sample_mask,
                    labels=sample_labels,
                    adapt=True
                )

                outputs_list.append(outputs)
                adaptation_infos.append(info)

            # Combine outputs
            combined_outputs = self._combine_outputs(outputs_list)

        else:
            # Adapt to entire batch at once
            outputs, info = self.adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                adapt=True
            )

            adaptation_infos = [info] * batch_size
            combined_outputs = outputs

        return combined_outputs, adaptation_infos

    def _combine_outputs(self, outputs_list: List[Any]) -> Any:
        """Combine outputs from multiple forward passes"""
        # This is a simplified version - real implementation would handle
        # different output types properly
        if hasattr(outputs_list[0], 'logits'):
            combined_logits = torch.cat([out.logits for out in outputs_list], dim=0)
            # Return a namedtuple or similar with combined outputs
            return type('Outputs', (), {'logits': combined_logits})()

        return outputs_list[0]

    def reset(self):
        """Reset adapter state"""
        self.adapter.reset()
