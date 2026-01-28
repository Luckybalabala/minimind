"""
Configuration Module for MiniMind

This module provides configuration classes for different training paradigms,
including Nested Learning.
"""

from .nested_config import (
    NestedLearningConfig,
    create_default_config,
    create_mvp_config,
    create_full_config,
)

__all__ = [
    'NestedLearningConfig',
    'create_default_config',
    'create_mvp_config',
    'create_full_config',
]
