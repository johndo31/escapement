"""
Trace data structures for capturing agent execution state.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types of steps that can occur in an agent trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    USER_INPUT = "user_input"
    AGENT_OUTPUT = "agent_output"
    ERROR = "error"
    CHECKPOINT = "checkpoint"


class LLMRequest(BaseModel):
    """Captured LLM API request."""

    provider: str = "openai"
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    # Additional parameters captured as-is
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute a deterministic hash for cache lookup."""
        # Normalize the request for consistent hashing
        normalized = {
            "provider": self.provider,
            "model": self.model,
            "messages": self._normalize_messages(self.messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self._normalize_json(self.tools) if self.tools else None,
            "tool_choice": self._normalize_json(self.tool_choice) if self.tool_choice else None,
            "seed": self.seed,
        }
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages for consistent hashing."""
        normalized = []
        for msg in messages:
            norm_msg = {
                "role": msg.get("role", ""),
                "content": self._normalize_content(msg.get("content", "")),
            }
            if "name" in msg:
                norm_msg["name"] = msg["name"]
            if "tool_calls" in msg:
                norm_msg["tool_calls"] = self._normalize_json(msg["tool_calls"])
            if "tool_call_id" in msg:
                norm_msg["tool_call_id"] = msg["tool_call_id"]
            normalized.append(norm_msg)
        return normalized

    def _normalize_content(self, content: Any) -> Any:
        """Normalize content, stripping excess whitespace."""
        if isinstance(content, str):
            return " ".join(content.split())
        return content

    def _normalize_json(self, obj: Any) -> Any:
        """Recursively normalize JSON for consistent hashing."""
        if isinstance(obj, dict):
            return {k: self._normalize_json(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [self._normalize_json(item) for item in obj]
        return obj


class LLMResponse(BaseModel):
    """Captured LLM API response."""

    id: str | None = None
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None = None
    created: int | None = None
    # Full raw response for complete fidelity
    raw_response: dict[str, Any] | None = None


class ToolRequest(BaseModel):
    """Captured tool/function call."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str | None = None


class ToolResponse(BaseModel):
    """Captured tool/function response."""

    result: Any
    error: str | None = None
    duration_ms: float | None = None


class TraceStep(BaseModel):
    """A single step in an agent execution trace."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    step_number: int
    step_type: StepType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = None

    # LLM-specific fields
    llm_request: LLMRequest | None = None
    llm_response: LLMResponse | None = None
    request_hash: str | None = None

    # Tool-specific fields
    tool_request: ToolRequest | None = None
    tool_response: ToolResponse | None = None

    # Generic data for other step types
    data: dict[str, Any] = Field(default_factory=dict)

    # Local state snapshot (optional, for deep debugging)
    local_state: dict[str, Any] | None = None

    # Error information
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Compute request hash after initialization."""
        if self.llm_request and not self.request_hash:
            self.request_hash = self.llm_request.compute_hash()


class TraceMetadata(BaseModel):
    """Metadata about a trace."""

    trace_id: str
    name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    status: str = "running"  # running, completed, failed, replaying
    total_steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    tags: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    agent_info: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """Complete trace of an agent execution."""

    metadata: TraceMetadata
    steps: list[TraceStep] = Field(default_factory=list)

    # Replay state
    replay_mode: bool = False
    replay_cursor: int = 0
    fork_point: int | None = None  # Step number where we forked from replay to live

    @classmethod
    def create(cls, name: str | None = None, tags: list[str] | None = None) -> Trace:
        """Create a new trace."""
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        return cls(
            metadata=TraceMetadata(
                trace_id=trace_id,
                name=name,
                tags=tags or [],
            )
        )

    @property
    def trace_id(self) -> str:
        return self.metadata.trace_id

    def add_step(self, step: TraceStep) -> None:
        """Add a step to the trace."""
        step.step_number = len(self.steps)
        self.steps.append(step)
        self.metadata.total_steps = len(self.steps)

        # Update token counts if this was an LLM call
        if step.llm_response and step.llm_response.usage:
            usage = step.llm_response.usage
            self.metadata.total_tokens += usage.get("total_tokens", 0)

    def get_step(self, step_number: int) -> TraceStep | None:
        """Get a specific step by number."""
        if 0 <= step_number < len(self.steps):
            return self.steps[step_number]
        return None

    def complete(self, status: str = "completed") -> None:
        """Mark the trace as complete."""
        self.metadata.status = status
        self.metadata.ended_at = datetime.now(timezone.utc)

    def get_llm_steps(self) -> list[TraceStep]:
        """Get all LLM call steps."""
        return [s for s in self.steps if s.step_type == StepType.LLM_CALL]

    def get_tool_steps(self) -> list[TraceStep]:
        """Get all tool call steps."""
        return [s for s in self.steps if s.step_type == StepType.TOOL_CALL]

    def find_step_by_hash(self, request_hash: str) -> TraceStep | None:
        """Find an LLM step by its request hash (for replay cache lookup)."""
        for step in self.steps:
            if step.request_hash == request_hash:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Deserialize trace from dictionary."""
        return cls.model_validate(data)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Trace: {self.trace_id}",
            f"Status: {self.metadata.status}",
            f"Steps: {self.metadata.total_steps}",
            f"Tokens: {self.metadata.total_tokens}",
            f"LLM Calls: {len(self.get_llm_steps())}",
            f"Tool Calls: {len(self.get_tool_steps())}",
        ]
        if self.metadata.name:
            lines.insert(1, f"Name: {self.metadata.name}")
        return "\n".join(lines)
