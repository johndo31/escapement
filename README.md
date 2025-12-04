# Escapement

**Deterministic replay engine for AI agents.** Debug Heisenbugs with time-travel debugging.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

AI agents are non-deterministic. The same input can produce different outputs. This makes debugging nearly impossible:

- **Heisenbugs**: Bugs that disappear when you try to reproduce them
- **Token burn**: Testing costs real money ($2-20 per failed debug session)
- **Infinite loops**: Runaway agents that burn through your API budget overnight

## The Solution

Escapement captures and replays agent executions deterministically. Record once, replay forever, at zero cost.

```python
import escapement
from openai import OpenAI

# Start recording
escapement.init(name="customer-support-agent")

# Your agent runs normally
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Help me debug this"}]
)

# Stop recording
escapement.stop()
```

Later, replay the exact execution:

```python
import escapement

# Replay uses cached responses - $0 cost
escapement.replay("tr_abc123def456")

# Your agent code runs identically
# Same prompts, same responses, same bugs
```

## Features

- **One-line integration**: `import escapement; escapement.init()`
- **Zero-cost replay**: Cached responses mean free regression testing
- **Loop detection**: Automatically kills runaway agents before they burn your budget
- **Time-travel debugging**: Fork from any point and go live
- **Multi-provider**: Works with OpenAI, Anthropic, OpenRouter, and any OpenAI-compatible API

## Supported Providers

| Provider | Status | Notes |
|----------|--------|-------|
| OpenAI | Supported | Full support including tool calls |
| Anthropic | Supported | Claude models, tool use |
| OpenRouter | Supported | Uses OpenAI SDK, works automatically |
| Azure OpenAI | Supported | Uses OpenAI SDK, works automatically |
| LangChain | Supported | Via callback handler |

## Installation

```bash
# Core (no provider SDKs)
pip install escapement

# With OpenAI
pip install escapement[openai]

# With Anthropic
pip install escapement[anthropic]

# With everything
pip install escapement[all]
```

## Quick Start

### 1. Record an agent execution

```python
import escapement
from openai import OpenAI

trace = escapement.init(name="my-agent", tags=["production"])

client = OpenAI()
# ... your agent code ...

escapement.stop()
print(f"Recorded trace: {trace.trace_id}")
```

### 2. List recorded traces

```bash
$ escapement list

┌─────────────────┬────────────────────┬───────────┬───────┬────────┬──────────────────┐
│ Trace ID        │ Name               │ Status    │ Steps │ Tokens │ Created          │
├─────────────────┼────────────────────┼───────────┼───────┼────────┼──────────────────┤
│ tr_abc123def456 │ my-agent           │ completed │    12 │   4521 │ 2025-01-15 14:30 │
│ tr_xyz789ghi012 │ customer-support   │ failed    │     8 │   2103 │ 2025-01-15 13:45 │
└─────────────────┴────────────────────┴───────────┴───────┴────────┴──────────────────┘
```

### 3. Replay a trace

```python
import escapement

# Load the trace - all LLM calls will be served from cache
escapement.replay("tr_abc123def456")

# Run your agent code - it will follow the exact same path
# ...
```

### 4. Time-travel debugging

```python
import escapement

# Start in replay mode
escapement.replay("tr_abc123def456")

# Run until the bug...
# ... agent code ...

# Fork to live mode to try a fix
escapement.fork()

# Now LLM calls go to the real API
# Test your fix without re-running setup steps
```

## Provider Examples

### Anthropic

```python
import escapement
from anthropic import Anthropic

escapement.init(name="claude-agent")

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude!"}]
)

escapement.stop()
```

### OpenRouter

```python
import escapement
from openai import OpenAI

escapement.init(name="openrouter-agent")

# OpenRouter uses OpenAI SDK with different base_url
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key",
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)

escapement.stop()
# Works automatically - no extra configuration needed
```

## CLI Commands

```bash
# List all traces
escapement list

# Show trace details
escapement show tr_abc123def456

# Show with message content
escapement show tr_abc123def456 --content

# Export trace to JSON (for sharing)
escapement export tr_abc123def456

# Import a shared trace
escapement import trace.json

# Delete a trace
escapement delete tr_abc123def456

# Show statistics
escapement stats tr_abc123def456

# Show configuration
escapement info
```

## Configuration

Environment variables:

```bash
# Storage location (default: ./.escapement)
export ESCAPEMENT_STORAGE_DIR=/path/to/traces

# Enable/disable recording (default: true)
export ESCAPEMENT_ENABLED=true

# Max identical calls before loop detection (default: 10)
export ESCAPEMENT_MAX_IDENTICAL_CALLS=10
```

Programmatic configuration:

```python
from escapement import EscapementConfig, init

config = EscapementConfig(
    storage_dir="/custom/path",
    loop_detection_enabled=True,
    max_identical_calls=5,
    strict_replay=True,
)

init(config=config)
```

## LangChain Integration

```python
import escapement
from escapement.integrations import EscapementCallbackHandler
from langchain_openai import ChatOpenAI

# Initialize escapement
escapement.init(name="langchain-agent")

# Create handler
handler = EscapementCallbackHandler()

# Attach to LLM
llm = ChatOpenAI(callbacks=[handler])

# Your LangChain code runs normally
# All calls are automatically traced
```

## How It Works

### Recording

When you call `escapement.init()`, we install lightweight interceptors on LLM client libraries (OpenAI, Anthropic). Every API call is:

1. **Hashed**: We compute a deterministic hash of the request (model, messages, parameters)
2. **Recorded**: Request and response are stored in a local SQLite database
3. **Timed**: We capture latency for performance analysis

### Replay

When you call `escapement.replay(trace_id)`, we:

1. **Load** the recorded trace
2. **Intercept** LLM calls as before
3. **Match** each request to its cached response by hash
4. **Return** the cached response instantly (no API call, no cost)

### Loop Detection

We track recent request hashes. If the same request appears N times (default: 10), we:

1. **Kill** the agent immediately
2. **Save** the trace for debugging
3. **Raise** `LoopDetectedError` with the trace ID

This prevents runaway agents from burning through your API budget.

## Limitations

- **New prompts require live calls**: Replay only works for identical requests. If you change a prompt, that call goes live.
- **External state not captured**: If your agent reads from a database or API, those calls aren't mocked (yet).
- **Streaming responses**: Currently aggregated. Timing simulation coming soon.

## Roadmap

- [x] Anthropic Claude support
- [x] OpenRouter support (via OpenAI SDK)
- [ ] Tool/function call mocking
- [ ] Streaming replay with timing simulation
- [ ] VS Code extension for visual debugging
- [ ] Cloud sync for team collaboration
- [ ] Auto-generate regression tests from traces

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

---

**Why "Escapement"?** In mechanical watches, the escapement is the mechanism that controls the release of energy, turning chaotic spring tension into precise, measured ticks. Escapement does the same for AI agents: turning chaotic, non-deterministic LLM calls into reproducible, debuggable executions.
