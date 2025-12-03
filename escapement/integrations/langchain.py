"""
LangChain integration for Escapement.

Provides a callback handler that automatically captures LangChain
agent executions as Escapement traces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore


class EscapementCallbackHandler(BaseCallbackHandler if LANGCHAIN_AVAILABLE else object):
    """
    LangChain callback handler for Escapement tracing.

    This handler automatically captures LangChain LLM calls, tool
    invocations, and chain executions as Escapement trace steps.

    Usage:
        from langchain_openai import ChatOpenAI
        from escapement.integrations import EscapementCallbackHandler

        handler = EscapementCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler])

        # Or attach to all chains:
        import escapement
        escapement.init(name="my-agent")

        from langchain.globals import set_llm_cache
        # ... your agent code
    """

    def __init__(self, trace_name: str | None = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install escapement[langchain]"
            )

        super().__init__()
        self.trace_name = trace_name
        self._run_id_to_step: Dict[UUID, int] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts processing."""
        from escapement.core import get_current_trace, checkpoint

        trace = get_current_trace()
        if trace:
            checkpoint(
                name="langchain_llm_start",
                data={
                    "run_id": str(run_id),
                    "model": serialized.get("name", "unknown"),
                    "prompts": prompts[:2],  # Truncate for storage
                },
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes processing."""
        from escapement.core import get_current_trace, checkpoint

        trace = get_current_trace()
        if trace:
            # Extract token usage if available
            token_usage = {}
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})

            checkpoint(
                name="langchain_llm_end",
                data={
                    "run_id": str(run_id),
                    "generations": len(response.generations),
                    "token_usage": token_usage,
                },
            )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts execution."""
        from escapement.core import get_current_trace, checkpoint

        trace = get_current_trace()
        if trace:
            checkpoint(
                name="langchain_tool_start",
                data={
                    "run_id": str(run_id),
                    "tool_name": serialized.get("name", "unknown"),
                    "input_preview": input_str[:500] if input_str else None,
                },
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes execution."""
        from escapement.core import record_tool_call

        # Record as a proper tool call for replay purposes
        record_tool_call(
            tool_name=f"langchain_tool_{run_id}",
            arguments={"run_id": str(run_id)},
            result=str(output)[:1000] if output else None,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        from escapement.core import record_tool_call

        record_tool_call(
            tool_name=f"langchain_tool_{run_id}",
            arguments={"run_id": str(run_id)},
            result=None,
            error=str(error),
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        from escapement.core import get_current_trace, checkpoint

        trace = get_current_trace()
        if trace:
            checkpoint(
                name="langchain_chain_start",
                data={
                    "run_id": str(run_id),
                    "chain_type": serialized.get("name", "unknown"),
                },
            )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends."""
        from escapement.core import get_current_trace, checkpoint

        trace = get_current_trace()
        if trace:
            checkpoint(
                name="langchain_chain_end",
                data={
                    "run_id": str(run_id),
                    "output_keys": list(outputs.keys()) if outputs else [],
                },
            )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        from escapement.core import get_current_trace, checkpoint

        trace = get_current_trace()
        if trace:
            checkpoint(
                name="langchain_chain_error",
                data={
                    "run_id": str(run_id),
                    "error": str(error),
                },
            )
