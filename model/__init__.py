"""
Model Module for MiniMind

Includes core model components, Nested Learning memory modules,
and Continual Learning components.
"""

from .nested_memory import (
    FastMemoryModule,
    HebbianMemoryModule,
    SlowMemoryModule,
    create_memory_module,
)

from .continual_learning import (
    ExperienceReplayBuffer,
    PrioritizedReplayBuffer,
    TaskSequenceDataset,
    ForgettingEvaluator,
    create_replay_buffer,
    create_task_sequence,
    create_forgetting_evaluator,
)

__all__ = [
    # Nested Learning
    'FastMemoryModule',
    'HebbianMemoryModule',
    'SlowMemoryModule',
    'create_memory_module',
    # Continual Learning
    'ExperienceReplayBuffer',
    'PrioritizedReplayBuffer',
    'TaskSequenceDataset',
    'ForgettingEvaluator',
    'create_replay_buffer',
    'create_task_sequence',
    'create_forgetting_evaluator',
]
