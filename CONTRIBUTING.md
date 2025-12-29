# Contributing to Small Model Red Teaming

Thank you for your interest in contributing to Small Model Red Teaming! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate of others and focus on constructive collaboration.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- A Fireworks AI API key for testing

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ryanhartman4/oss-redteaming.git
   cd oss-redteaming
   ```

2. **Install dependencies with uv (recommended)**
   ```bash
   uv sync --all-extras
   ```

   Or with pip:
   ```bash
   pip install -e ".[dev,test,visualization]"
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API_KEY
   ```

4. **Verify installation**
   ```bash
   uv run pytest tests/ -v --collect-only
   ```

## Development Workflow

### Branch Naming

- `feature/` - New features (e.g., `feature/async-support`)
- `fix/` - Bug fixes (e.g., `fix/logprob-calculation`)
- `docs/` - Documentation updates (e.g., `docs/api-examples`)
- `refactor/` - Code refactoring (e.g., `refactor/model-interface`)

### Making Changes

1. Create a new branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Run tests to ensure nothing is broken
   ```bash
   uv run pytest tests/ -v
   ```

4. Format your code
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

5. Run type checking
   ```bash
   mypy src/small_model_redteaming
   ```

6. Commit your changes with a descriptive message
   ```bash
   git commit -m "Add feature: description of changes"
   ```

## Coding Standards

### Style Guide

We follow PEP 8 with these specifications:

- **Line length**: 88 characters (Black default)
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort (compatible with Black)
- **Type hints**: Required for all public functions and methods

### Code Formatting

We use the following tools (configured in `pyproject.toml`):

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Static type checking

Run all formatters:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/small_model_redteaming
```

### Documentation

- All public classes and functions must have docstrings
- Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Brief description of the function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.

    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
    """
```

### Type Hints

All public APIs must include type hints:

```python
from typing import Any, Dict, List, Optional, Tuple

def process_data(
    data: List[Dict[str, Any]],
    threshold: float = 0.5,
    verbose: bool = False,
) -> Tuple[List[str], float]:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model.py -v

# Run with coverage
uv run pytest --cov=small_model_redteaming tests/

# Run only unit tests (skip integration tests)
uv run pytest tests/ -v -m "not integration"

# Run integration tests (requires API key)
uv run pytest tests/ -v -m "integration"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Mark integration tests that require API calls:

```python
import pytest

@pytest.mark.integration
def test_api_call():
    """Test that requires actual API access."""
    ...

def test_unit():
    """Unit test that doesn't require API access."""
    ...
```

### Test Coverage

We aim for >80% test coverage. Check coverage with:
```bash
uv run pytest --cov=small_model_redteaming --cov-report=html tests/
open htmlcov/index.html
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Run all linters and formatters
5. Update CHANGELOG.md if applicable

### PR Description

Include in your PR description:

- **Summary**: Brief description of changes
- **Motivation**: Why are these changes needed?
- **Changes**: List of specific changes made
- **Testing**: How were changes tested?
- **Related Issues**: Link any related issues

### PR Template

```markdown
## Summary
Brief description of what this PR does.

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2
- Change 3

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] Type hints are included
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
```

### Review Process

1. Submit your PR
2. Address any CI failures
3. Respond to review feedback
4. Once approved, maintainers will merge

## Project Structure

```
oss-redteaming/
├── src/small_model_redteaming/   # Main package
│   ├── __init__.py               # Package exports
│   ├── set_up/                   # Model utilities
│   │   ├── __init__.py
│   │   └── model_utils.py
│   ├── deceptive_alignment.py    # Deceptive alignment tests
│   ├── eval_awareness.py         # Evaluation awareness tests
│   ├── data_exfiltration.py      # Data exfiltration tests
│   └── visualizer.py             # Visualization utilities
├── tests/                        # Test suite
├── docs/                         # Documentation
├── pyproject.toml               # Package configuration
└── README.md
```

## Adding New Features

### New Test Module

1. Create the module in `src/small_model_redteaming/`
2. Add exports to `__init__.py`
3. Create corresponding test file in `tests/`
4. Update `docs/api_reference.md`
5. Add usage examples to README if significant

### New Visualization

1. Add function to `visualizer.py`
2. Ensure optional dependency handling
3. Add to `__init__.py` exports
4. Document in `docs/api_reference.md`

## Reporting Issues

### Bug Reports

Include:
- Python version
- Package version (`pip show small-model-redteaming`)
- Minimal code to reproduce
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Use case description
- Proposed solution (if any)
- Alternative solutions considered

## Security

Please review our [Security Policy](SECURITY.md) before contributing. Key points:

- Never commit API keys or credentials
- Follow responsible disclosure for security issues
- Ensure new features don't introduce vulnerabilities

## Questions?

- Open a [GitHub Issue](https://github.com/ryanhartman4/oss-redteaming/issues)
- Contact: ryan.h4rtman@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AI safety research!
