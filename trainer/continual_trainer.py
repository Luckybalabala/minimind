"""
Continual Learning Trainer for MiniMind with Nested Learning

Implements task-based training loop with experience replay and memory consolidation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json
import time

from model.continual_learning import (
    ExperienceReplayBuffer,
    TaskSequenceDataset,
    ForgettingEvaluator,
)
from trainer.nested_optimizer import NestedOptimizer


class ContinualTrainer:
    """
    Trainer for Continual Learning with Nested Learning

    Coordinates multi-task training with experience replay, nested optimization,
    and forgetting evaluation.

    Attributes:
        model: The model being trained
        task_sequence: Task sequence dataset
        replay_buffer: Experience replay buffer
        evaluator: Forgetting evaluator
        nested_optimizer: Nested optimizer for multi-timescale learning
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        task_sequence: TaskSequenceDataset,
        replay_buffer: ExperienceReplayBuffer,
        evaluator: ForgettingEvaluator,
        base_lr: float = 1e-4,
        device: str = "cuda",
        save_dir: str = "../checkpoints"
    ):
        """
        Initialize Continual Trainer

        Args:
            model: PyTorch model
            config: Model configuration
            task_sequence: Task sequence dataset
            replay_buffer: Experience replay buffer
            evaluator: Forgetting evaluator
            base_lr: Base learning rate
            device: Training device
            save_dir: Checkpoint directory
        """
        self.model = model.to(device)
        self.config = config
        self.task_sequence = task_sequence
        self.replay_buffer = replay_buffer
        self.evaluator = evaluator
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create nested optimizer
        if config.use_nested_learning:
            self.optimizer = NestedOptimizer(
                model=model,
                config=config,
                base_optimizer_cls=optim.AdamW,
                base_lr=base_lr
            )
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)

        # Training state
        self.current_task_id = None
        self.global_step = 0
        self.task_steps = {}  # Steps per task

        # Training history
        self.history = {
            'task_losses': [],
            'replay_losses': [],
            'evaluations': []
        }

    def train_task(
        self,
        task_id: int,
        epochs: int,
        dataloader: Optional[Any] = None,
        replay_freq: int = 100,
        eval_freq: int = 500,
        log_freq: int = 10
    ):
        """
        Train on a single task

        Args:
            task_id: Task to train on
            epochs: Number of epochs
            dataloader: Optional data loader (if not provided, uses task_sequence)
            replay_freq: Frequency of experience replay
            eval_freq: Frequency of evaluation
            log_freq: Frequency of logging
        """
        self.current_task_id = task_id
        self.task_steps[task_id] = 0

        print(f"\n{'='*70}")
        print(f"Training on Task {task_id}")
        print(f"{'='*70}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Training loop
            if dataloader is None:
                # Use task sequence to get batches
                for step in range(100):  # Arbitrary number of steps
                    batch = self.task_sequence.get_task_batch(task_id, batch_size=8)
                    loss = self._train_step(batch, task_id)

                    epoch_loss += loss
                    num_batches += 1

                    # Experience replay
                    if step % replay_freq == 0:
                        replay_loss = self._replay_step(replay_freq)
                        if replay_loss is not None:
                            print(f"  Step {step}: Replay Loss = {replay_loss:.4f}")

                    # Logging
                    if step % log_freq == 0:
                        print(f"  Epoch {epoch+1}/{epochs}, Step {step}: Loss = {loss:.4f}")

            else:
                # Use provided dataloader
                for step, batch in enumerate(dataloader):
                    loss = self._train_step(batch, task_id)

                    epoch_loss += loss
                    num_batches += 1

                    # Experience replay
                    if step % replay_freq == 0:
                        self._replay_step(replay_freq)

                    if step % log_freq == 0:
                        print(f"  Epoch {epoch+1}/{epochs}, Step {step}: Loss = {loss:.4f}")

            # Evaluation
            if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
                avg_accuracy = self._evaluate_on_all_tasks()
                print(f"  Epoch {epoch+1}: Average Accuracy = {avg_accuracy:.4f}")

            # Average loss for epoch
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"  Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Task completion
        self._consolidate_memory(task_id)
        print(f"✅ Task {task_id} completed")

    def _train_step(self, batch: Any, task_id: int) -> float:
        """
        Single training step

        Args:
            batch: Training batch
            task_id: Current task ID

        Returns:
            Loss value
        """
        self.model.train()

        # Forward pass
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
        else:
            # Assume batch is already tensors
            input_ids = batch.to(self.device)
            labels = batch.to(self.device)

        # Forward
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss

        # Backward
        if self.config.use_nested_learning:
            # Use nested optimizer
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Nested optimizer step
            self.optimizer.step(loss)
            self.optimizer.zero_grad()
        else:
            # Standard optimizer
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        # Store experience
        if isinstance(batch, (tuple, list)):
            experience = (input_ids.cpu(), labels.cpu())
        else:
            experience = batch.cpu()

        self.replay_buffer.add(task_id, experience, loss=loss.item())

        self.global_step += 1
        self.task_steps[task_id] += 1

        return loss.item()

    def _replay_step(self, num_samples: int) -> Optional[float]:
        """
        Experience replay step

        Args:
            num_samples: Number of replay samples

        Returns:
            Replay loss if buffer has data, None otherwise
        """
        if self.replay_buffer.get_total_size() == 0:
            return None

        # Sample replay batch
        replay_batch = self.task_sequence.get_replay_batch(
            self.replay_buffer,
            num_samples,
            strategy="forgetting_curve"
        )

        if len(replay_batch) == 0:
            return None

        # Train on replay batch
        replay_loss = 0.0
        for task_id, experience in replay_batch:
            if isinstance(experience, (tuple, list)):
                input_ids, labels = experience
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
            else:
                input_ids = experience.to(self.device)
                labels = experience.to(self.device)

            # Forward and backward
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

            if self.config.use_nested_learning:
                loss.backward()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            replay_loss += loss.item()

        return replay_loss / len(replay_batch) if len(replay_batch) > 0 else 0.0

    def _evaluate_on_all_tasks(self) -> float:
        """
        Evaluate on all seen tasks

        Returns:
            Average accuracy across all tasks
        """
        # This would need actual evaluation logic
        # For now, return placeholder
        accuracies = []

        for task_id in self.task_sequence.task_order:
            if len(self.task_sequence.task_stats.get(task_id, {})) > 0:
                # Use last accuracy if available
                stats = self.task_sequence.task_stats[task_id]
                if 'accuracies' in stats and len(stats['accuracies']) > 0:
                    accuracies.append(stats['accuracies'][-1])

        return sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0

    def _consolidate_memory(self, task_id: int):
        """
        Consolidate memory after task completion

        Moves knowledge from fast memory to slower memory levels.

        Args:
            task_id: Completed task ID
        """
        print(f"  🔄 Consolidating memory for Task {task_id}...")

        # In a full implementation, this would:
        # 1. Extract important patterns from fast memory
        # 2. Merge into medium/long-term memory
        # 3. Possibly prune less important connections

        # For now, this is a placeholder
        pass

    def train_continual(
        self,
        num_epochs_per_task: int = 5,
        replay_freq: int = 100,
        eval_freq: int = 1
    ):
        """
        Train on sequence of tasks

        Args:
            num_epochs_per_task: Epochs to train each task
            replay_freq: Frequency of experience replay
            eval_freq: Frequency of evaluation
        """
        print(f"\n{'='*70}")
        print("Starting Continual Learning Training")
        print(f"Tasks: {self.task_sequence.task_order}")
        print(f"Total Tasks: {len(self.task_sequence.tasks)}")
        print(f"{'='*70}")

        # Initial evaluation
        initial_avg_acc = self._evaluate_on_all_tasks()
        print(f"\nInitial Average Accuracy: {initial_avg_acc:.4f}")

        # Train each task sequentially
        for task_id in self.task_sequence.task_order:
            self.train_task(
                task_id=task_id,
                epochs=num_epochs_per_task,
                replay_freq=replay_freq,
                eval_freq=eval_freq
            )

            # Advance to next task
            if not self.task_sequence.advance_task():
                break

        # Final evaluation
        final_avg_acc = self._evaluate_on_all_tasks()
        final_report = self.evaluator.generate_report()

        print(f"\n{'='*70}")
        print("Continual Learning Training Complete")
        print(f"{'='*70}")
        print(f"\nFinal Average Accuracy: {final_avg_acc:.4f}")
        print(f"Average Forgetting Rate: {final_report['average_forgetting_rate']:.4f}")

        # Print detailed report
        self.evaluator.print_report()

        # Save training history
        self.save_history()

    def save_history(self, filepath: Optional[str] = None):
        """Save training history"""
        if filepath is None:
            filepath = self.save_dir / "continual_learning_history.json"

        history = {
            'global_step': self.global_step,
            'task_steps': self.task_steps,
            'history': self.history,
            'evaluator_report': self.evaluator.generate_report()
        }

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n💾 Training history saved to {filepath}")

    def save_checkpoint(self, filepath: Optional[str] = None):
        """Save model and trainer state"""
        if filepath is None:
            filepath = self.save_dir / "continual_trainer_checkpoint.pth"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None,
            'global_step': self.global_step,
            'current_task_id': self.current_task_id,
            'task_steps': self.task_steps,
            'evaluator_state': {
                'task_performance': self.evaluator.task_performance,
                'task_names': self.evaluator.task_names
            }
        }

        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model and trainer state"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_task_id = checkpoint['current_task_id']
        self.task_steps = checkpoint['task_steps']

        if 'optimizer' in checkpoint and checkpoint['optimizer']:
            if hasattr(self.optimizer, 'load_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'evaluator_state' in checkpoint:
            self.evaluator.task_performance = {
                int(k): v for k, v in checkpoint['evaluator_state']['task_performance'].items()
            }
            self.evaluator.task_names = {
                int(k): v for k, v in checkpoint['evaluator_state']['task_names'].items()
            }

        print(f"✅ Checkpoint loaded from {filepath}")


def create_continual_trainer(
    model: nn.Module,
    config: Any,
    task_sequence: TaskSequenceDataset,
    replay_buffer: ExperienceReplayBuffer,
    evaluator: ForgettingEvaluator,
    **kwargs
) -> ContinualTrainer:
    """
    Factory function to create continual trainer

    Args:
        model: PyTorch model
        config: Model configuration
        task_sequence: Task sequence dataset
        replay_buffer: Experience replay buffer
        evaluator: Forgetting evaluator
        **kwargs: Additional arguments

    Returns:
        ContinualTrainer instance
    """
    return ContinualTrainer(
        model=model,
        config=config,
        task_sequence=task_sequence,
        replay_buffer=replay_buffer,
        evaluator=evaluator,
        **kwargs
    )
