"""
Core Escapement functionality - init, replay, and state management.
"""

from __future__ import annotations

import atexit
import contextvars
from typing import Any

from escapement.config import EscapementConfig, get_config, set_config
from escapement.interceptor import install_interceptors, uninstall_interceptors
from escapement.storage import get_storage
from escapement.tracer import Trace

# Context variables for thread-safe state management
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "escapement_current_trace", default=None
)
_replay_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "escapement_replay_trace", default=None
)
_replay_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "escapement_replay_mode", default=False
)
_initialized: bool = False


def init(
    name: str | None = None,
    tags: list[str] | None = None,
    config: EscapementConfig | None = None,
    auto_save: bool = True,
) -> Trace:
    """
    Initialize Escapement and start a new trace.

    This is the main entry point. Call this once at the start of your
    agent execution to begin recording.

    Args:
        name: Optional name for the trace
        tags: Optional list of tags for filtering
        config: Optional configuration override
        auto_save: Whether to save the trace on exit

    Returns:
        The newly created Trace object

    Example:
        >>> import escapement
        >>> trace = escapement.init(name="customer-support-agent")
        >>> # Your agent code runs here...
        >>> escapement.stop()
    """
    global _initialized

    # Apply configuration
    if config:
        set_config(config)

    cfg = get_config()
    cfg.ensure_storage_dir()

    # Create new trace
    trace = Trace.create(name=name, tags=tags)
    _current_trace.set(trace)
    _replay_mode.set(False)

    # Install interceptors
    install_interceptors()

    # Save initial trace
    storage = get_storage()
    storage.save_trace(trace)

    # Register cleanup on exit
    if auto_save and not _initialized:
        atexit.register(_cleanup)

    _initialized = True

    return trace


def stop(status: str = "completed") -> Trace | None:
    """
    Stop recording and finalize the current trace.

    Args:
        status: Final status of the trace (completed, failed, cancelled)

    Returns:
        The completed Trace object, or None if no trace was active
    """
    trace = _current_trace.get()
    if trace is None:
        return None

    # Finalize trace
    trace.complete(status=status)

    # Save final state
    storage = get_storage()
    storage.save_trace(trace)

    # Clear state
    _current_trace.set(None)

    return trace


def replay(
    trace_id: str,
    strict: bool = True,
    fork_at: int | None = None,
) -> Trace | None:
    """
    Enter replay mode using a saved trace.

    In replay mode, LLM calls that match recorded requests will return
    cached responses instead of making live API calls.

    Args:
        trace_id: ID of the trace to replay
        strict: If True, raise error on cache miss. If False, fall through to live API.
        fork_at: Optional step number to fork at (switch from replay to live)

    Returns:
        The loaded Trace object, or None if not found

    Example:
        >>> import escapement
        >>> escapement.replay("tr_abc123")
        >>> # Your agent code runs, but uses cached responses
    """
    storage = get_storage()
    trace = storage.load_trace(trace_id)

    if trace is None:
        return None

    # Set replay state
    trace.replay_mode = True
    trace.fork_point = fork_at
    _replay_trace.set(trace)
    _replay_mode.set(True)

    # Create a new trace to record the replay
    cfg = get_config()
    cfg.strict_replay = strict

    replay_trace = Trace.create(
        name=f"replay-of-{trace_id}",
        tags=["replay", f"source:{trace_id}"],
    )
    replay_trace.replay_mode = True
    _current_trace.set(replay_trace)

    # Install interceptors
    install_interceptors()

    return trace


def fork() -> None:
    """
    Fork from replay mode to live mode.

    After calling this, subsequent LLM calls will go to the live API
    instead of serving cached responses. Useful for "time travel"
    debugging where you want to modify behavior mid-execution.
    """
    _replay_mode.set(False)

    trace = _current_trace.get()
    if trace:
        trace.fork_point = len(trace.steps)


def get_current_trace() -> Trace | None:
    """Get the currently active trace."""
    return _current_trace.get()


def get_replay_trace() -> Trace | None:
    """Get the trace being used as replay source."""
    return _replay_trace.get()


def is_replay_mode() -> bool:
    """Check if we're currently in replay mode."""
    return _replay_mode.get()


def _cleanup() -> None:
    """Cleanup handler called at program exit."""
    trace = _current_trace.get()
    if trace and trace.metadata.status == "running":
        trace.complete(status="interrupted")
        storage = get_storage()
        storage.save_trace(trace)


# Convenience functions for manual step recording


def record_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    error: str | None = None,
    duration_ms: float | None = None,
) -> None:
    """
    Manually record a tool call in the current trace.

    Use this when you have custom tools that aren't automatically
    intercepted.

    Args:
        tool_name: Name of the tool/function called
        arguments: Arguments passed to the tool
        result: Return value from the tool
        error: Error message if the tool failed
        duration_ms: How long the tool call took
    """
    from escapement.tracer import StepType, ToolRequest, ToolResponse, TraceStep

    trace = _current_trace.get()
    if trace is None:
        return

    step = TraceStep(
        step_type=StepType.TOOL_CALL,
        tool_request=ToolRequest(
            tool_name=tool_name,
            arguments=arguments,
        ),
        tool_response=ToolResponse(
            result=result,
            error=error,
            duration_ms=duration_ms,
        ),
        duration_ms=duration_ms,
    )
    trace.add_step(step)

    storage = get_storage()
    storage.save_step(trace.trace_id, step)


def checkpoint(name: str, data: dict[str, Any] | None = None) -> None:
    """
    Record a checkpoint in the trace.

    Checkpoints are useful for marking significant points in agent
    execution, like "started task" or "finished planning phase".

    Args:
        name: Name of the checkpoint
        data: Optional data to attach
    """
    from escapement.tracer import StepType, TraceStep

    trace = _current_trace.get()
    if trace is None:
        return

    step = TraceStep(
        step_type=StepType.CHECKPOINT,
        data={"name": name, **(data or {})},
    )
    trace.add_step(step)

    storage = get_storage()
    storage.save_step(trace.trace_id, step)
