# Contributing to Escapement

Thanks for your interest in contributing to Escapement!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/johndo31/escapement.git
cd escapement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

We use `black` for formatting and `ruff` for linting:

```bash
black escapement/
ruff check escapement/
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new functionality
3. Ensure tests pass and code is formatted
4. Update README.md if you've changed public APIs
5. Submit PR with a clear description of changes

## Adding a New Provider

To add support for a new LLM provider:

1. Create an interceptor class in `escapement/interceptor.py` following the pattern of `OpenAIInterceptor` or `AnthropicInterceptor`
2. Add the provider SDK as an optional dependency in `pyproject.toml`
3. Register the interceptor in `install_interceptors()` and `uninstall_interceptors()`
4. Add an example in `examples/`
5. Update README.md with the new provider

Key methods to implement:
- `install()` / `uninstall()` - Monkey-patch the SDK
- `_build_request()` - Convert provider's request format to `LLMRequest`
- `_build_response()` - Convert provider's response format to `LLMResponse`
- `_response_to_X_object()` - Convert `LLMResponse` back to provider's format for replay

## Reporting Issues

- Check existing issues first
- Include Python version, OS, and provider SDK versions
- Include minimal reproduction code if possible

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
