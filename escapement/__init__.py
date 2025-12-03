"""
Escapement: Deterministic Replay Engine for AI Agents

Usage:
    import escapement
    escapement.init()  # Start recording all LLM interactions

    # Your agent code runs normally...

    # Later, replay a specific trace:
    escapement.replay("trace_abc123")
"""

from escapement.core import init, replay, get_current_trace, stop
from escapement.tracer import Trace, TraceStep, StepType
from escapement.config import EscapementConfig

__version__ = "0.1.0"
__all__ = [
    "init",
    "replay",
    "stop",
    "get_current_trace",
    "Trace",
    "TraceStep",
    "StepType",
    "EscapementConfig",
]
