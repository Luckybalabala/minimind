"""
Continual Learning Components for MiniMind

This module implements experience replay, task management, and forgetting
evaluation for continual learning with Nested Learning.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import deque
import random
import math


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for Continual Learning

    Implements a multi-task experience replay buffer with various sampling strategies,
    including FOREVER-style forgetting curve sampling.

    Attributes:
        buffer_size: Maximum number of experiences per task
        num_tasks: Number of tasks to track
        sampling_strategy: Strategy for sampling experiences
        forgetting_curve: Whether to use forgetting curve weighting
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        num_tasks: int = 5,
        sampling_strategy: str = "diverse",
        use_forgetting_curve: bool = True,
        decay_rate: float = 0.5
    ):
        """
        Initialize Experience Replay Buffer

        Args:
            buffer_size: Max experiences per task
            num_tasks: Number of tasks to support
            sampling_strategy: 'uniform', 'diverse', 'recent', or 'forgetting_curve'
            use_forgetting_curve: Enable FOREVER-style forgetting curve
            decay_rate: Decay rate for forgetting curve
        """
        self.buffer_size = buffer_size
        self.num_tasks = num_tasks
        self.sampling_strategy = sampling_strategy
        self.use_forgetting_curve = use_forgetting_curve
        self.decay_rate = decay_rate

        # Create separate buffer for each task
        self.buffers: Dict[int, deque] = {
            task_id: deque(maxlen=buffer_size)
            for task_id in range(num_tasks)
        }

        # Track metadata for each experience
        self.metadata: Dict[int, Dict] = {
            task_id: {}  # experience_id -> {'timestamp': int, 'age': int}
            for task_id in range(num_tasks)
        }

        # Task sizes
        self.task_sizes: Dict[int, int] = {task_id: 0 for task_id in range(num_tasks)}

        # Current timestamp
        self.current_timestamp = 0

    def add(self, task_id: int, experience: Any, metadata: Optional[Dict] = None):
        """
        Add an experience to the buffer

        Args:
            task_id: Task identifier
            experience: Experience data (can be tensors, dict, etc.)
            metadata: Optional metadata (loss, difficulty, etc.)
        """
        if task_id not in self.buffers:
            self.buffers[task_id] = deque(maxlen=self.buffer_size)
            self.metadata[task_id] = {}

        # Add to buffer
        exp_id = len(self.metadata[task_id])
        self.buffers[task_id].append((exp_id, experience))

        # Store metadata
        meta = {
            'timestamp': self.current_timestamp,
            'task_id': task_id,
        }
        if metadata:
            meta.update(metadata)

        self.metadata[task_id][exp_id] = meta
        self.task_sizes[task_id] = len(self.buffers[task_id])

        # Increment timestamp
        self.current_timestamp += 1

    def sample(self, batch_size: int, strategy: Optional[str] = None) -> List[Tuple[int, Any]]:
        """
        Sample experiences from the buffer

        Args:
            batch_size: Number of experiences to sample
            strategy: Override default sampling strategy

        Returns:
            List of (task_id, experience) tuples
        """
        if strategy is None:
            strategy = self.sampling_strategy

        if strategy == "uniform":
            return self._sample_uniform(batch_size)
        elif strategy == "diverse":
            return self._sample_diverse(batch_size)
        elif strategy == "recent":
            return self._sample_recent(batch_size)
        elif strategy == "forgetting_curve":
            return self._sample_forgetting_curve(batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _sample_uniform(self, batch_size: int) -> List[Tuple[int, Any]]:
        """Uniform random sampling across all tasks"""
        samples = []

        # Collect all experiences
        all_experiences = []
        for task_id, buffer in self.buffers.items():
            for exp_id, exp in buffer:
                all_experiences.append((task_id, exp_id, exp))

        if len(all_experiences) == 0:
            return samples

        # Sample uniformly
        indices = random.sample(range(len(all_experiences)), min(batch_size, len(all_experiences)))
        samples = [(all_experiences[i][0], all_experiences[i][2]) for i in indices]

        return samples

    def _sample_diverse(self, batch_size: int) -> List[Tuple[int, Any]]:
        """Diverse sampling - ensure samples from multiple tasks"""
        samples = []

        # Get active tasks (with data)
        active_tasks = [tid for tid, buf in self.buffers.items() if len(buf) > 0]

        if len(active_tasks) == 0:
            return samples

        # Calculate samples per task
        samples_per_task = max(1, batch_size // len(active_tasks))

        for task_id in active_tasks:
            buffer = self.buffers[task_id]
            n_samples = min(samples_per_task, len(buffer))

            # Sample from this task
            exp_ids = random.sample(range(len(buffer)), n_samples)
            for exp_id in exp_ids:
                _, exp = buffer[exp_id]
                samples.append((task_id, exp))

        # If we need more samples, add randomly
        if len(samples) < batch_size:
            remaining = batch_size - len(samples)
            extra_samples = self._sample_uniform(remaining)
            samples.extend(extra_samples)

        return samples

    def _sample_recent(self, batch_size: int) -> List[Tuple[int, Any]]:
        """Sample recent experiences (higher priority to recent)"""
        samples = []

        # Collect all experiences with timestamps
        all_experiences = []
        for task_id, buffer in self.buffers.items():
            for exp_id, exp in buffer:
                timestamp = self.metadata[task_id][exp_id]['timestamp']
                all_experiences.append((timestamp, task_id, exp))

        if len(all_experiences) == 0:
            return samples

        # Sort by timestamp (newest first)
        all_experiences.sort(key=lambda x: x[0], reverse=True)

        # Sample from recent portion
        recent_portion = all_experiences[:min(len(all_experiences), batch_size * 2)]
        indices = random.sample(range(len(recent_portion)), min(batch_size, len(recent_portion)))

        samples = [(recent_portion[i][1], recent_portion[i][2]) for i in indices]

        return samples

    def _sample_forgetting_curve(self, batch_size: int) -> List[Tuple[int, Any]]:
        """
        Sample based on forgetting curve (FOREVER-style)

        Priority given to experiences that are more likely to be forgotten.
        Uses Ebbinghaus forgetting curve: R(t) = exp(-t / decay_rate)
        """
        samples = []

        # Collect all experiences with forgetting weights
        weighted_experiences = []
        current_time = self.current_timestamp

        for task_id, buffer in self.buffers.items():
            for exp_id, exp in buffer:
                # Calculate age
                timestamp = self.metadata[task_id][exp_id]['timestamp']
                age = current_time - timestamp

                # Calculate forgetting weight (higher = more likely to be sampled)
                if age == 0:
                    weight = 1.0
                else:
                    # Inverse of forgetting curve (sample those more likely forgotten)
                    weight = 1.0 - math.exp(-age / self.decay_rate)

                weighted_experiences.append((weight, task_id, exp))

        if len(weighted_experiences) == 0:
            return samples

        # Sample based on weights
        weights = [w for w, _, _ in weighted_experiences]
        total_weight = sum(weights)

        if total_weight > 0:
            # Normalize weights
            probs = [w / total_weight for w in weights]

            # Sample
            indices = np.random.choice(
                len(weighted_experiences),
                size=min(batch_size, len(weighted_experiences)),
                replace=False,
                p=probs
            )

            samples = [(weighted_experiences[i][1], weighted_experiences[i][2]) for i in indices]

        return samples

    def get_task_buffer_size(self, task_id: int) -> int:
        """Get current buffer size for a specific task"""
        return self.task_sizes.get(task_id, 0)

    def get_total_size(self) -> int:
        """Get total number of experiences across all tasks"""
        return sum(self.task_sizes.values())

    def clear_task(self, task_id: int):
        """Clear buffer for a specific task"""
        if task_id in self.buffers:
            self.buffers[task_id].clear()
            self.metadata[task_id].clear()
            self.task_sizes[task_id] = 0

    def clear_all(self):
        """Clear all buffers"""
        for task_id in self.buffers:
            self.buffers[task_id].clear()
            self.metadata[task_id].clear()
            self.task_sizes[task_id] = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'total_experiences': self.get_total_size(),
            'current_timestamp': self.current_timestamp,
            'task_sizes': dict(self.task_sizes),
            'active_tasks': len([s for s in self.task_sizes.values() if s > 0])
        }


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    """
    Prioritized Experience Replay Buffer

    Samples experiences based on their priority (typically TD-error or loss).
    More useful experiences are sampled more frequently.
    """

    def __init__(self, *args, alpha: float = 0.6, **kwargs):
        """
        Initialize Prioritized Replay Buffer

        Args:
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.priorities: Dict[int, Dict[int, float]] = {
            task_id: {} for task_id in self.buffers.keys()
        }

    def add(self, task_id: int, experience: Any, priority: Optional[float] = None, **kwargs):
        """Add experience with priority"""
        super().add(task_id, experience, metadata=kwargs)

        exp_id = len(self.metadata[task_id]) - 1

        # Set priority (max priority for new experiences)
        if priority is None:
            priority = 1.0

        self.priorities[task_id][exp_id] = priority

    def sample(self, batch_size: int, **kwargs) -> List[Tuple[int, Any]]:
        """Sample based on priorities"""
        samples = []

        # Collect all experiences with priorities
        all_experiences = []
        for task_id, buffer in self.buffers.items():
            for exp_id, exp in buffer:
                priority = self.priorities[task_id].get(exp_id, 1.0)
                all_experiences.append((priority, task_id, exp))

        if len(all_experiences) == 0:
            return samples

        # Sample based on priorities
        priorities = [p ** self.alpha for p, _, _ in all_experiences]
        total_priority = sum(priorities)

        if total_priority > 0:
            probs = [p / total_priority for p in priorities]
            indices = np.random.choice(
                len(all_experiences),
                size=min(batch_size, len(all_experiences)),
                replace=False,
                p=probs
            )
            samples = [(all_experiences[i][1], all_experiences[i][2]) for i in indices]

        return samples

    def update_priorities(self, task_id: int, exp_ids: List[int], priorities: List[float]):
        """Update priorities for experiences"""
        for exp_id, priority in zip(exp_ids, priorities):
            if exp_id in self.priorities[task_id]:
                self.priorities[task_id][exp_id] = priority


def create_replay_buffer(
    buffer_type: str = "standard",
    buffer_size: int = 10000,
    num_tasks: int = 5,
    **kwargs
) -> ExperienceReplayBuffer:
    """
    Factory function to create replay buffer

    Args:
        buffer_type: 'standard' or 'prioritized'
        buffer_size: Max experiences per task
        num_tasks: Number of tasks
        **kwargs: Additional arguments

    Returns:
        Experience replay buffer instance
    """
    if buffer_type == "prioritized":
        return PrioritizedReplayBuffer(
            buffer_size=buffer_size,
            num_tasks=num_tasks,
            **kwargs
        )
    else:
        return ExperienceReplayBuffer(
            buffer_size=buffer_size,
            num_tasks=num_tasks,
            **kwargs
        )


class TaskSequenceDataset:
    """
    Task Sequence Dataset for Continual Learning

    Manages a sequence of tasks for continual learning, with support for
    task switching, difficulty assessment, and statistics tracking.

    Attributes:
        tasks: Dictionary of task datasets
        task_order: Order of task presentation
        current_task: Current active task
        task_stats: Statistics for each task
    """

    def __init__(self, task_order: Optional[List[int]] = None):
        """
        Initialize Task Sequence Dataset

        Args:
            task_order: Optional list specifying task order
        """
        self.tasks: Dict[int, Any] = {}
        self.task_order = task_order if task_order is not None else []
        self.current_task_idx = 0
        self.task_stats: Dict[int, Dict] = {}

        # Task difficulty scores
        self.task_difficulties: Dict[int, float] = {}

    def add_task(self, task_id: int, task_data: Any, difficulty: Optional[float] = None):
        """
        Add a task to the sequence

        Args:
            task_id: Task identifier
            task_data: Task dataset/data loader
            difficulty: Optional difficulty score (0-1)
        """
        self.tasks[task_id] = task_data

        if task_id not in self.task_order:
            self.task_order.append(task_id)

        if difficulty is not None:
            self.task_difficulties[task_id] = difficulty

        # Initialize stats
        self.task_stats[task_id] = {
            'samples_seen': 0,
            'total_loss': 0.0,
            'epochs_trained': 0
        }

    def get_task_batch(self, task_id: int, batch_size: int, **kwargs) -> Any:
        """
        Get a batch from a specific task

        Args:
            task_id: Task identifier
            batch_size: Batch size
            **kwargs: Additional arguments for data loader

        Returns:
            Batch of data from the task
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task_data = self.tasks[task_id]

        # Handle different types of task data
        if hasattr(task_data, 'get_batch'):
            # Custom data loader
            batch = task_data.get_batch(batch_size, **kwargs)
        elif isinstance(task_data, list):
            # List of samples
            indices = random.sample(range(len(task_data)), min(batch_size, len(task_data)))
            batch = [task_data[i] for i in indices]
        elif hasattr(task_data, '__len__') and hasattr(task_data, '__getitem__'):
            # Array-like dataset
            indices = random.sample(range(len(task_data)), min(batch_size, len(task_data)))
            batch = [task_data[i] for i in indices]
        else:
            raise TypeError(f"Unsupported task data type: {type(task_data)}")

        # Update stats
        self.task_stats[task_id]['samples_seen'] += len(batch) if isinstance(batch, list) else batch_size

        return batch

    def get_replay_batch(self, replay_buffer: ExperienceReplayBuffer, batch_size: int, **kwargs) -> List[Tuple[int, Any]]:
        """
        Get a replay batch from the experience replay buffer

        Args:
            replay_buffer: Experience replay buffer
            batch_size: Batch size
            **kwargs: Arguments for sampling strategy

        Returns:
            List of (task_id, experience) tuples
        """
        return replay_buffer.sample(batch_size, **kwargs)

    def get_current_task(self) -> Optional[int]:
        """Get current task ID"""
        if self.current_task_idx < len(self.task_order):
            return self.task_order[self.current_task_idx]
        return None

    def advance_task(self):
        """Advance to next task in sequence"""
        if self.current_task_idx < len(self.task_order) - 1:
            self.current_task_idx += 1
            return True
        return False

    def reset_task_sequence(self):
        """Reset to first task"""
        self.current_task_idx = 0

    def get_task_difficulty(self, task_id: int) -> float:
        """Get task difficulty score"""
        return self.task_difficulties.get(task_id, 0.5)

    def evaluate_task(self, model: nn.Module, task_id: int, eval_fn: callable, **kwargs) -> Dict[str, float]:
        """
        Evaluate model on a specific task

        Args:
            model: Model to evaluate
            task_id: Task to evaluate on
            eval_fn: Evaluation function (model, data) -> metrics
            **kwargs: Additional arguments

        Returns:
            Dictionary of evaluation metrics
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        # Get test data for task
        task_data = self.tasks[task_id]

        # Evaluate
        metrics = eval_fn(model, task_data, **kwargs)

        # Update stats
        if 'accuracy' in metrics:
            self.task_stats[task_id].setdefault('accuracies', [])
            self.task_stats[task_id]['accuracies'].append(metrics['accuracy'])

        return metrics

    def get_all_task_stats(self) -> Dict[int, Dict]:
        """Get statistics for all tasks"""
        return self.task_stats.copy()

    def get_task_sequence_info(self) -> Dict[str, Any]:
        """Get information about the task sequence"""
        return {
            'total_tasks': len(self.tasks),
            'task_order': self.task_order.copy(),
            'current_task': self.get_current_task(),
            'task_difficulties': self.task_difficulties.copy()
        }

    def create_curriculum_order(self, strategy: str = "easy_first") -> List[int]:
        """
        Create a curriculum-based task order

        Args:
            strategy: 'easy_first', 'hard_first', 'alternating', or 'random'

        Returns:
            List of task IDs in curriculum order
        """
        if not self.task_difficulties:
            # No difficulty info, return original order
            return self.task_order.copy()

        if strategy == "easy_first":
            return sorted(self.task_order, key=lambda tid: self.get_task_difficulty(tid))
        elif strategy == "hard_first":
            return sorted(self.task_order, key=lambda tid: self.get_task_difficulty(tid), reverse=True)
        elif strategy == "alternating":
            # Alternate between easy and hard
            easy_tasks = sorted(self.task_order, key=lambda tid: self.get_task_difficulty(tid))
            hard_tasks = sorted(self.task_order, key=lambda tid: self.get_task_difficulty(tid), reverse=True)
            curriculum = []
            for e, h in zip(easy_tasks, hard_tasks):
                curriculum.extend([e, h])
            # Add remaining tasks
            for tid in self.task_order:
                if tid not in curriculum:
                    curriculum.append(tid)
            return curriculum
        elif strategy == "random":
            order = self.task_order.copy()
            random.shuffle(order)
            return order
        else:
            raise ValueError(f"Unknown curriculum strategy: {strategy}")


def create_task_sequence(
    tasks: Dict[int, Any],
    task_order: Optional[List[int]] = None
) -> TaskSequenceDataset:
    """
    Factory function to create task sequence dataset

    Args:
        tasks: Dictionary of task_id -> task_data
        task_order: Optional task order

    Returns:
        TaskSequenceDataset instance
    """
    dataset = TaskSequenceDataset(task_order)

    for task_id, task_data in tasks.items():
        dataset.add_task(task_id, task_data)

    return dataset


class ForgettingEvaluator:
    """
    Evaluator for measuring forgetting in continual learning

    Computes various metrics for catastrophic forgetting and continual learning performance.

    Attributes:
        task_performance: Performance history for each task
        num_tasks: Number of tasks being tracked
    """

    def __init__(self, num_tasks: int):
        """
        Initialize Forgetting Evaluator

        Args:
            num_tasks: Number of tasks to track
        """
        self.num_tasks = num_tasks
        self.task_performance: Dict[int, List[float]] = {
            task_id: [] for task_id in range(num_tasks)
        }
        self.task_names: Dict[int, str] = {}

    def set_task_name(self, task_id: int, name: str):
        """Set a human-readable name for a task"""
        self.task_names[task_id] = name

    def track_task_performance(self, task_id: int, accuracy: float, step: Optional[int] = None):
        """
        Track performance on a specific task

        Args:
            task_id: Task identifier
            accuracy: Accuracy or other performance metric
            step: Optional training step number
        """
        if task_id not in self.task_performance:
            self.task_performance[task_id] = []

        self.task_performance[task_id].append(accuracy)

    def compute_forgetting_rate(self, task_id: int) -> float:
        """
        Compute forgetting rate for a specific task

        Forgetting rate = (max_accuracy - current_accuracy) / max_accuracy

        Args:
            task_id: Task to compute forgetting for

        Returns:
            Forgetting rate (0 = no forgetting, 1 = complete forgetting)
        """
        if task_id not in self.task_performance or len(self.task_performance[task_id]) == 0:
            return 0.0

        performances = self.task_performance[task_id]
        max_perf = max(performances)
        current_perf = performances[-1]

        if max_perf == 0:
            return 0.0

        forgetting_rate = (max_perf - current_perf) / max_perf
        return max(0.0, forgetting_rate)

    def compute_average_accuracy(self) -> float:
        """
        Compute average accuracy across all tasks

        Uses the most recent performance for each task.

        Returns:
            Average accuracy
        """
        accuracies = []
        for task_id in range(self.num_tasks):
            if task_id in self.task_performance and len(self.task_performance[task_id]) > 0:
                accuracies.append(self.task_performance[task_id][-1])

        if len(accuracies) == 0:
            return 0.0

        return sum(accuracies) / len(accuracies)

    def compute_backward_transfer(self) -> Dict[int, float]:
        """
        Compute backward transfer (how much new tasks hurt old tasks)

        For each task, compares first performance to current performance.

        Returns:
            Dictionary of task_id -> backward_transfer_score
        """
        backward_transfers = {}

        for task_id in range(self.num_tasks):
            if task_id not in self.task_performance or len(self.task_performance[task_id]) < 2:
                backward_transfers[task_id] = 0.0
                continue

            performances = self.task_performance[task_id]
            first_perf = performances[0]
            current_perf = performances[-1]

            # Negative transfer means current < first (worse)
            backward_transfer = (current_perf - first_perf) / first_perf if first_perf > 0 else 0.0
            backward_transfers[task_id] = backward_transfer

        return backward_transfers

    def compute_forward_transfer(self) -> Dict[int, float]:
        """
        Compute forward transfer (how much previous tasks help new tasks)

        Compares each task's first performance to baseline.

        Returns:
            Dictionary of task_id -> forward_transfer_score
        """
        forward_transfers = {}

        for task_id in range(self.num_tasks):
            if task_id not in self.task_performance or len(self.task_performance[task_id]) == 0:
                forward_transfers[task_id] = 0.0
                continue

            # This would require baseline performance
            # For now, we'll use 0 as placeholder
            forward_transfers[task_id] = 0.0

        return forward_transfers

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report

        Returns:
            Dictionary with all metrics and statistics
        """
        report = {
            'average_accuracy': self.compute_average_accuracy(),
            'tasks': {}
        }

        # Per-task metrics
        for task_id in range(self.num_tasks):
            task_name = self.task_names.get(task_id, f"Task_{task_id}")

            if task_id in self.task_performance and len(self.task_performance[task_id]) > 0:
                performances = self.task_performance[task_id]

                report['tasks'][task_id] = {
                    'name': task_name,
                    'current_accuracy': performances[-1],
                    'max_accuracy': max(performances),
                    'min_accuracy': min(performances),
                    'mean_accuracy': sum(performances) / len(performances),
                    'forgetting_rate': self.compute_forgetting_rate(task_id),
                    'num_evaluations': len(performances)
                }
            else:
                report['tasks'][task_id] = {
                    'name': task_name,
                    'status': 'Not evaluated'
                }

        # Overall metrics
        forgetting_rates = [self.compute_forgetting_rate(tid) for tid in range(self.num_tasks)]
        report['average_forgetting_rate'] = sum(forgetting_rates) / len(forgetting_rates) if forgetting_rates else 0.0

        backward_transfers = self.compute_backward_transfer()
        if backward_transfers:
            report['average_backward_transfer'] = sum(backward_transfers.values()) / len(backward_transfers)

        return report

    def print_report(self):
        """Print a formatted evaluation report"""
        report = self.generate_report()

        print("\n" + "=" * 70)
        print("Continual Learning Evaluation Report")
        print("=" * 70)

        print(f"\n📊 Overall Metrics:")
        print(f"  Average Accuracy: {report['average_accuracy']:.4f}")
        print(f"  Average Forgetting Rate: {report['average_forgetting_rate']:.4f}")

        if 'average_backward_transfer' in report:
            print(f"  Average Backward Transfer: {report['average_backward_transfer']:.4f}")

        print(f"\n📋 Per-Task Metrics:")
        for task_id, task_report in report['tasks'].items():
            if task_report.get('status') == 'Not evaluated':
                continue

            print(f"\n  {task_report['name']} (Task {task_id}):")
            print(f"    Current Accuracy:  {task_report['current_accuracy']:.4f}")
            print(f"    Max Accuracy:      {task_report['max_accuracy']:.4f}")
            print(f"    Forgetting Rate:    {task_report['forgetting_rate']:.4f}")

        print("\n" + "=" * 70 + "\n")

    def plot_learning_curves(self, save_path: Optional[str] = None):
        """
        Plot learning curves for all tasks

        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))

            # Plot each task's performance over time
            for task_id in range(self.num_tasks):
                if task_id not in self.task_performance or len(self.task_performance[task_id]) == 0:
                    continue

                performances = self.task_performance[task_id]
                task_name = self.task_names.get(task_id, f"Task {task_id}")

                plt.plot(performances, marker='o', label=task_name)

            plt.xlabel('Evaluation Step')
            plt.ylabel('Accuracy')
            plt.title('Continual Learning - Task Performance Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")

            plt.close()

        except ImportError:
            print("Warning: matplotlib not available, skipping plot")

    def get_forgetting_curve(self, task_id: int) -> List[float]:
        """
        Get the forgetting curve for a specific task

        Args:
            task_id: Task to get curve for

        Returns:
            List of performance values over time
        """
        if task_id not in self.task_performance:
            return []

        return self.task_performance[task_id].copy()

    def save(self, filepath: str):
        """Save evaluator state to file"""
        import json

        state = {
            'num_tasks': self.num_tasks,
            'task_performance': self.task_performance,
            'task_names': self.task_names
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: str):
        """Load evaluator state from file"""
        import json

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.num_tasks = state['num_tasks']
        self.task_performance = {
            int(k): v for k, v in state['task_performance'].items()
        }
        self.task_names = {
            int(k): v for k, v in state['task_names'].items()
        }


def create_forgetting_evaluator(num_tasks: int) -> ForgettingEvaluator:
    """
    Factory function to create forgetting evaluator

    Args:
        num_tasks: Number of tasks to track

    Returns:
        ForgettingEvaluator instance
    """
    return ForgettingEvaluator(num_tasks)
