"""
Interceptors for LLM API calls.

This module provides monkey-patching functionality to intercept
LLM API calls transparently, enabling recording and replay.

Supported providers:
- OpenAI (and OpenAI-compatible APIs like OpenRouter, Azure OpenAI, etc.)
- Anthropic

OpenRouter Note:
    OpenRouter uses the OpenAI SDK with a different base_url, so it's
    automatically intercepted by the OpenAI interceptor. No additional
    configuration needed.
"""

from __future__ import annotations

import functools
import time
from collections import deque
from typing import Any, Callable

from escapement.config import get_config
from escapement.tracer import (
    LLMRequest,
    LLMResponse,
    StepType,
    Trace,
    TraceStep,
)


class LoopDetectedError(Exception):
    """Raised when an infinite loop is detected in agent behavior."""

    def __init__(self, message: str, trace_id: str, repeated_hash: str):
        super().__init__(message)
        self.trace_id = trace_id
        self.repeated_hash = repeated_hash


class ReplayCacheMissError(Exception):
    """Raised when replay mode cannot find a cached response."""

    def __init__(self, message: str, request_hash: str):
        super().__init__(message)
        self.request_hash = request_hash


class OpenAIInterceptor:
    """
    Intercepts OpenAI API calls for recording and replay.

    This class wraps the OpenAI client's chat.completions.create method
    to capture requests and responses, and optionally serve cached
    responses during replay mode.
    """

    def __init__(self):
        self._original_create: Callable[..., Any] | None = None
        self._original_async_create: Callable[..., Any] | None = None
        self._installed = False
        self._recent_hashes: deque[str] = deque(maxlen=100)

    def install(self) -> None:
        """Install the interceptor by monkey-patching the OpenAI client."""
        if self._installed:
            return

        try:
            import openai
            from openai.resources.chat import completions
        except ImportError:
            return  # OpenAI not installed, skip

        # Store original methods
        self._original_create = completions.Completions.create
        self._original_async_create = completions.AsyncCompletions.create

        # Create wrapped versions
        interceptor = self

        @functools.wraps(self._original_create)
        def wrapped_create(self_client: Any, *args: Any, **kwargs: Any) -> Any:
            return interceptor._intercept_call(
                self_client, interceptor._original_create, *args, **kwargs
            )

        @functools.wraps(self._original_async_create)
        async def wrapped_async_create(self_client: Any, *args: Any, **kwargs: Any) -> Any:
            return await interceptor._intercept_async_call(
                self_client, interceptor._original_async_create, *args, **kwargs
            )

        # Monkey-patch
        completions.Completions.create = wrapped_create  # type: ignore
        completions.AsyncCompletions.create = wrapped_async_create  # type: ignore

        self._installed = True

    def uninstall(self) -> None:
        """Remove the interceptor and restore original methods."""
        if not self._installed:
            return

        try:
            from openai.resources.chat import completions
        except ImportError:
            return

        if self._original_create:
            completions.Completions.create = self._original_create  # type: ignore
        if self._original_async_create:
            completions.AsyncCompletions.create = self._original_async_create  # type: ignore

        self._installed = False

    def _intercept_call(
        self,
        client: Any,
        original_fn: Callable[..., Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercept a synchronous API call."""
        from escapement.core import get_current_trace, is_replay_mode, get_replay_trace

        config = get_config()
        current_trace = get_current_trace()

        # Build request object
        request = self._build_request(kwargs)
        request_hash = request.compute_hash()

        # Check for loops
        if config.loop_detection_enabled and current_trace:
            self._check_for_loop(request_hash, current_trace)

        # Replay mode: serve from cache
        if is_replay_mode():
            replay_trace = get_replay_trace()
            if replay_trace:
                cached_step = replay_trace.find_step_by_hash(request_hash)
                if cached_step and cached_step.llm_response:
                    return self._response_to_openai_object(cached_step.llm_response)

                if config.strict_replay:
                    raise ReplayCacheMissError(
                        f"No cached response for request hash {request_hash}",
                        request_hash,
                    )
                # Fall through to live call if not strict

        # Make the actual API call
        if original_fn is None:
            raise RuntimeError("Original function not available")

        start_time = time.perf_counter()
        response = original_fn(client, *args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record the call
        if current_trace and config.enabled:
            llm_response = self._build_response(response)
            step = TraceStep(
                step_type=StepType.LLM_CALL,
                llm_request=request,
                llm_response=llm_response,
                request_hash=request_hash,
                duration_ms=duration_ms,
            )
            current_trace.add_step(step)

            # Persist step
            from escapement.storage import get_storage

            get_storage().save_step(current_trace.trace_id, step)

        return response

    async def _intercept_async_call(
        self,
        client: Any,
        original_fn: Callable[..., Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercept an asynchronous API call."""
        from escapement.core import get_current_trace, is_replay_mode, get_replay_trace

        config = get_config()
        current_trace = get_current_trace()

        # Build request object
        request = self._build_request(kwargs)
        request_hash = request.compute_hash()

        # Check for loops
        if config.loop_detection_enabled and current_trace:
            self._check_for_loop(request_hash, current_trace)

        # Replay mode: serve from cache
        if is_replay_mode():
            replay_trace = get_replay_trace()
            if replay_trace:
                cached_step = replay_trace.find_step_by_hash(request_hash)
                if cached_step and cached_step.llm_response:
                    return self._response_to_openai_object(cached_step.llm_response)

                if config.strict_replay:
                    raise ReplayCacheMissError(
                        f"No cached response for request hash {request_hash}",
                        request_hash,
                    )

        # Make the actual API call
        if original_fn is None:
            raise RuntimeError("Original function not available")

        start_time = time.perf_counter()
        response = await original_fn(client, *args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record the call
        if current_trace and config.enabled:
            llm_response = self._build_response(response)
            step = TraceStep(
                step_type=StepType.LLM_CALL,
                llm_request=request,
                llm_response=llm_response,
                request_hash=request_hash,
                duration_ms=duration_ms,
            )
            current_trace.add_step(step)

            from escapement.storage import get_storage

            get_storage().save_step(current_trace.trace_id, step)

        return response

    def _build_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        """Build an LLMRequest from API call kwargs."""
        # Extract known parameters
        known_params = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
        }

        extra_params = {k: v for k, v in kwargs.items() if k not in known_params}

        return LLMRequest(
            provider="openai",
            model=kwargs.get("model", "unknown"),
            messages=kwargs.get("messages", []),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            response_format=kwargs.get("response_format"),
            seed=kwargs.get("seed"),
            extra_params=extra_params,
        )

    def _build_response(self, response: Any) -> LLMResponse:
        """Build an LLMResponse from an OpenAI response object."""
        # Handle both Pydantic models and dicts
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif hasattr(response, "to_dict"):
            data = response.to_dict()
        else:
            data = dict(response) if hasattr(response, "__iter__") else {}

        return LLMResponse(
            id=getattr(response, "id", None),
            model=getattr(response, "model", "unknown"),
            choices=[
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (c.message.tool_calls or [])
                        ]
                        if c.message.tool_calls
                        else None,
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in response.choices
            ],
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
            created=getattr(response, "created", None),
            raw_response=data,
        )

    def _response_to_openai_object(self, llm_response: LLMResponse) -> Any:
        """Convert an LLMResponse back to an OpenAI-like response object."""
        # For replay, we return a mock object that behaves like the OpenAI response
        from types import SimpleNamespace

        def make_namespace(d: dict[str, Any]) -> SimpleNamespace:
            """Recursively convert dict to SimpleNamespace."""
            ns = SimpleNamespace()
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(ns, k, make_namespace(v))
                elif isinstance(v, list):
                    setattr(
                        ns,
                        k,
                        [make_namespace(item) if isinstance(item, dict) else item for item in v],
                    )
                else:
                    setattr(ns, k, v)
            return ns

        # Build a response-like object
        response_dict: dict[str, Any] = {
            "id": llm_response.id or "cached-response",
            "model": llm_response.model,
            "choices": [],
            "created": llm_response.created or 0,
        }

        for choice in llm_response.choices:
            message_dict: dict[str, Any] = {
                "role": choice["message"]["role"],
                "content": choice["message"]["content"],
            }

            if choice["message"].get("tool_calls"):
                message_dict["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in choice["message"]["tool_calls"]
                ]
            else:
                message_dict["tool_calls"] = None

            response_dict["choices"].append(
                {
                    "index": choice["index"],
                    "message": message_dict,
                    "finish_reason": choice["finish_reason"],
                }
            )

        if llm_response.usage:
            response_dict["usage"] = {
                "prompt_tokens": llm_response.usage.get("prompt_tokens", 0),
                "completion_tokens": llm_response.usage.get("completion_tokens", 0),
                "total_tokens": llm_response.usage.get("total_tokens", 0),
            }

        return make_namespace(response_dict)

    def _check_for_loop(self, request_hash: str, trace: Trace) -> None:
        """Check if we're in an infinite loop."""
        config = get_config()

        # Count occurrences of this hash in recent history
        self._recent_hashes.append(request_hash)
        count = sum(1 for h in self._recent_hashes if h == request_hash)

        if count >= config.max_identical_calls:
            trace.complete(status="failed_loop")
            raise LoopDetectedError(
                f"Detected infinite loop: identical request made {count} times. "
                f"Trace saved as {trace.trace_id}",
                trace.trace_id,
                request_hash,
            )


class AnthropicInterceptor:
    """
    Intercepts Anthropic API calls for recording and replay.

    This class wraps the Anthropic client's messages.create method
    to capture requests and responses, and optionally serve cached
    responses during replay mode.
    """

    def __init__(self):
        self._original_create: Callable[..., Any] | None = None
        self._original_async_create: Callable[..., Any] | None = None
        self._installed = False
        self._recent_hashes: deque[str] = deque(maxlen=100)

    def install(self) -> None:
        """Install the interceptor by monkey-patching the Anthropic client."""
        if self._installed:
            return

        try:
            from anthropic.resources import messages
        except ImportError:
            return  # Anthropic not installed, skip

        # Store original methods
        self._original_create = messages.Messages.create

        # Try to get async version if it exists
        try:
            from anthropic.resources import AsyncMessages
            self._original_async_create = AsyncMessages.create
        except (ImportError, AttributeError):
            pass

        # Create wrapped versions
        interceptor = self

        @functools.wraps(self._original_create)
        def wrapped_create(self_client: Any, *args: Any, **kwargs: Any) -> Any:
            return interceptor._intercept_call(
                self_client, interceptor._original_create, *args, **kwargs
            )

        # Monkey-patch sync
        messages.Messages.create = wrapped_create  # type: ignore

        # Monkey-patch async if available
        if self._original_async_create:
            @functools.wraps(self._original_async_create)
            async def wrapped_async_create(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                return await interceptor._intercept_async_call(
                    self_client, interceptor._original_async_create, *args, **kwargs
                )

            try:
                from anthropic.resources import AsyncMessages
                AsyncMessages.create = wrapped_async_create  # type: ignore
            except (ImportError, AttributeError):
                pass

        self._installed = True

    def uninstall(self) -> None:
        """Remove the interceptor and restore original methods."""
        if not self._installed:
            return

        try:
            from anthropic.resources import messages
        except ImportError:
            return

        if self._original_create:
            messages.Messages.create = self._original_create  # type: ignore

        if self._original_async_create:
            try:
                from anthropic.resources import AsyncMessages
                AsyncMessages.create = self._original_async_create  # type: ignore
            except (ImportError, AttributeError):
                pass

        self._installed = False

    def _intercept_call(
        self,
        client: Any,
        original_fn: Callable[..., Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercept a synchronous API call."""
        from escapement.core import get_current_trace, is_replay_mode, get_replay_trace

        config = get_config()
        current_trace = get_current_trace()

        # Build request object
        request = self._build_request(kwargs)
        request_hash = request.compute_hash()

        # Check for loops
        if config.loop_detection_enabled and current_trace:
            self._check_for_loop(request_hash, current_trace)

        # Replay mode: serve from cache
        if is_replay_mode():
            replay_trace = get_replay_trace()
            if replay_trace:
                cached_step = replay_trace.find_step_by_hash(request_hash)
                if cached_step and cached_step.llm_response:
                    return self._response_to_anthropic_object(cached_step.llm_response)

                if config.strict_replay:
                    raise ReplayCacheMissError(
                        f"No cached response for request hash {request_hash}",
                        request_hash,
                    )

        # Make the actual API call
        if original_fn is None:
            raise RuntimeError("Original function not available")

        start_time = time.perf_counter()
        response = original_fn(client, *args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record the call
        if current_trace and config.enabled:
            llm_response = self._build_response(response)
            step = TraceStep(
                step_type=StepType.LLM_CALL,
                llm_request=request,
                llm_response=llm_response,
                request_hash=request_hash,
                duration_ms=duration_ms,
            )
            current_trace.add_step(step)

            from escapement.storage import get_storage
            get_storage().save_step(current_trace.trace_id, step)

        return response

    async def _intercept_async_call(
        self,
        client: Any,
        original_fn: Callable[..., Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercept an asynchronous API call."""
        from escapement.core import get_current_trace, is_replay_mode, get_replay_trace

        config = get_config()
        current_trace = get_current_trace()

        # Build request object
        request = self._build_request(kwargs)
        request_hash = request.compute_hash()

        # Check for loops
        if config.loop_detection_enabled and current_trace:
            self._check_for_loop(request_hash, current_trace)

        # Replay mode: serve from cache
        if is_replay_mode():
            replay_trace = get_replay_trace()
            if replay_trace:
                cached_step = replay_trace.find_step_by_hash(request_hash)
                if cached_step and cached_step.llm_response:
                    return self._response_to_anthropic_object(cached_step.llm_response)

                if config.strict_replay:
                    raise ReplayCacheMissError(
                        f"No cached response for request hash {request_hash}",
                        request_hash,
                    )

        # Make the actual API call
        if original_fn is None:
            raise RuntimeError("Original function not available")

        start_time = time.perf_counter()
        response = await original_fn(client, *args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Record the call
        if current_trace and config.enabled:
            llm_response = self._build_response(response)
            step = TraceStep(
                step_type=StepType.LLM_CALL,
                llm_request=request,
                llm_response=llm_response,
                request_hash=request_hash,
                duration_ms=duration_ms,
            )
            current_trace.add_step(step)

            from escapement.storage import get_storage
            get_storage().save_step(current_trace.trace_id, step)

        return response

    def _build_request(self, kwargs: dict[str, Any]) -> LLMRequest:
        """Build an LLMRequest from Anthropic API call kwargs."""
        # Convert Anthropic messages format to standard format
        messages = kwargs.get("messages", [])

        # Anthropic uses "system" as a top-level param, not in messages
        system_prompt = kwargs.get("system")

        # Normalize messages
        normalized_messages = []
        if system_prompt:
            normalized_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            normalized_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        extra_params = {}
        known_params = {"model", "messages", "max_tokens", "temperature", "system", "tools", "tool_choice"}
        for k, v in kwargs.items():
            if k not in known_params:
                extra_params[k] = v

        return LLMRequest(
            provider="anthropic",
            model=kwargs.get("model", "unknown"),
            messages=normalized_messages,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            extra_params=extra_params,
        )

    def _build_response(self, response: Any) -> LLMResponse:
        """Build an LLMResponse from an Anthropic response object."""
        # Handle Anthropic's response format
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = {}

        # Extract content - Anthropic returns content as a list of blocks
        content_blocks = getattr(response, "content", [])
        text_content = ""
        tool_calls = None

        for block in content_blocks:
            if hasattr(block, "type"):
                if block.type == "text":
                    text_content += getattr(block, "text", "")
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": getattr(block, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", ""),
                            "arguments": str(getattr(block, "input", {})),
                        },
                    })

        # Build choices in OpenAI-compatible format
        choices = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content,
                "tool_calls": tool_calls,
            },
            "finish_reason": getattr(response, "stop_reason", "end_turn"),
        }]

        # Extract usage
        usage = None
        if hasattr(response, "usage"):
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
            }

        return LLMResponse(
            id=getattr(response, "id", None),
            model=getattr(response, "model", "unknown"),
            choices=choices,
            usage=usage,
            raw_response=data,
        )

    def _response_to_anthropic_object(self, llm_response: LLMResponse) -> Any:
        """Convert an LLMResponse back to an Anthropic-like response object."""
        from types import SimpleNamespace

        # Build content blocks
        content_blocks = []

        if llm_response.choices and llm_response.choices[0]["message"]["content"]:
            content_blocks.append(SimpleNamespace(
                type="text",
                text=llm_response.choices[0]["message"]["content"],
            ))

        # Add tool use blocks if present
        tool_calls = llm_response.choices[0]["message"].get("tool_calls") if llm_response.choices else None
        if tool_calls:
            for tc in tool_calls:
                content_blocks.append(SimpleNamespace(
                    type="tool_use",
                    id=tc["id"],
                    name=tc["function"]["name"],
                    input=tc["function"]["arguments"],
                ))

        # Build usage
        usage = None
        if llm_response.usage:
            usage = SimpleNamespace(
                input_tokens=llm_response.usage.get("prompt_tokens", 0),
                output_tokens=llm_response.usage.get("completion_tokens", 0),
            )

        # Build response object
        response = SimpleNamespace(
            id=llm_response.id or "cached-response",
            type="message",
            role="assistant",
            model=llm_response.model,
            content=content_blocks,
            stop_reason=llm_response.choices[0]["finish_reason"] if llm_response.choices else "end_turn",
            usage=usage,
        )

        return response

    def _check_for_loop(self, request_hash: str, trace: Trace) -> None:
        """Check if we're in an infinite loop."""
        config = get_config()

        self._recent_hashes.append(request_hash)
        count = sum(1 for h in self._recent_hashes if h == request_hash)

        if count >= config.max_identical_calls:
            trace.complete(status="failed_loop")
            raise LoopDetectedError(
                f"Detected infinite loop: identical request made {count} times. "
                f"Trace saved as {trace.trace_id}",
                trace.trace_id,
                request_hash,
            )


# Global interceptor instances
_openai_interceptor: OpenAIInterceptor | None = None
_anthropic_interceptor: AnthropicInterceptor | None = None


def get_openai_interceptor() -> OpenAIInterceptor:
    """Get or create the global OpenAI interceptor."""
    global _openai_interceptor
    if _openai_interceptor is None:
        _openai_interceptor = OpenAIInterceptor()
    return _openai_interceptor


def get_anthropic_interceptor() -> AnthropicInterceptor:
    """Get or create the global Anthropic interceptor."""
    global _anthropic_interceptor
    if _anthropic_interceptor is None:
        _anthropic_interceptor = AnthropicInterceptor()
    return _anthropic_interceptor


def install_interceptors() -> None:
    """Install all available interceptors."""
    get_openai_interceptor().install()
    get_anthropic_interceptor().install()


def uninstall_interceptors() -> None:
    """Uninstall all interceptors."""
    if _openai_interceptor:
        _openai_interceptor.uninstall()
    if _anthropic_interceptor:
        _anthropic_interceptor.uninstall()
