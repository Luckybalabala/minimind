"""
Model Module for MiniMind

Includes core model components, Nested Learning memory modules,
Continual Learning components, and Test-Time Scaling components.
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

from .test_time_scaling import (
    TestTimeAdapter,
    MemoryBasedAdapter,
    HybridAdapter,
    AdaptationConfig,
    create_test_time_adapter,
    TestTimeBatchAdapter,
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
    # Test-Time Scaling
    'TestTimeAdapter',
    'MemoryBasedAdapter',
    'HybridAdapter',
    'AdaptationConfig',
    'create_test_time_adapter',
    'TestTimeBatchAdapter',
]
