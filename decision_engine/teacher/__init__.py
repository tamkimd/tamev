# decision_engine/teacher/__init__.py
"""
Pluggable Teacher System for TAMEV Knowledge Distillation.
"""

from .agent_teacher import AgentTeacher, resolve_agent_identity
from .base import BaseTeacherModel, TeacherFactory, TeacherRegistry
from .cached_teacher import CachedTeacher
from .mock_teacher import MockTeacher

try:
    from .hf_teacher import HuggingFaceTeacher
except ImportError:
    HuggingFaceTeacher = None

try:
    from .api_teacher import APITeacher
except ImportError:
    APITeacher = None

__all__ = [
    "APITeacher",
    "AgentTeacher",
    "BaseTeacherModel",
    "CachedTeacher",
    "HuggingFaceTeacher",
    "MockTeacher",
    "TeacherFactory",
    "TeacherRegistry",
]
