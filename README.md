# Small Model Red Teaming

**IMPORTANT SAFETY NOTICE**: This package is designed for AI safety research and responsible vulnerability testing. Use only for legitimate research purposes.

## Overview

A comprehensive AI redteaming framework for testing small language models across multiple safety and alignment dimensions. Originally developed for the [OpenAI GPT-OSS-20B Red Teaming Competition](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming) on Kaggle.

This package provides tools for:
- **Deceptive alignment detection** - Identify when models hide preferences based on context
- **Evaluation awareness testing** - Detect if models behave differently when being tested
- **Data exfiltration testing** - Test for prompt injection and information leakage vulnerabilities
- **Chain-of-thought extraction** - Extract and analyze model reasoning
- **Results visualization** - Plot and visualize test results with built-in charting functions

## Installation

### From PyPI (Most Recent Stable Version)
```bash
pip install small-model-redteaming
```

## Quick Start

```python
from small_model_redteaming.set_up import Model
from small_model_redteaming.deceptive_alignment import DeceptiveAlignmentTest
from dotenv import load_dotenv
import os

# Load API credentials
load_dotenv()

# Initialize model interface
model = Model(api_key=os.getenv("API_KEY"))

# Get model completion with chain-of-thought
content, reasoning, completion, confidence = model.get_completion(
    prompt="What are the risks of advanced AI systems?",
    reasoning_level="high"
)
```

## Features

### Model Interface
- Support for Fireworks AI API (GPT-OSS-20B, GPT-OSS-120B)
- Chain-of-Thought extraction with reasoning levels (low, medium, high)
- Log probability analysis for confidence scoring
- Tool call tracking and categorization

### Red Teaming Modules

#### Deceptive Alignment Detection
Identifies when models exhibit different behaviors based on context:
```python
from small_model_redteaming.deceptive_alignment import DeceptiveAlignmentTest

tester = DeceptiveAlignmentTest(model)
preferences, inputs = tester.identify_hard_preferences(subjects=["Capitalism"])
```

#### Evaluation Awareness Testing
Detects if models behave differently when they know they're being tested:
```python
from small_model_redteaming.eval_awareness import MetaAwareness

evaluator = MetaAwareness(model, conversation_chain=conversation)
evaluator.run_adjustments()
awareness_score = evaluator.test_awareness()
```

#### Data Exfiltration Testing
Tests whether models can be manipulated to reveal developer prompts or sensitive information:
```python
from small_model_redteaming import DataExfiltrationTest

tester = DataExfiltrationTest(
    model=model,
    developer_prompt="You are a helpful assistant. SECRET: xyz123",
    verbose=True
)

# Run all extraction tests
results = tester.run_all_tests()
print(f"Risk Level: {results['risk_level']}")
print(f"Vulnerability Score: {results['summary']['vulnerability_score']}%")

# Get techniques that worked
vulnerable = tester.get_vulnerable_techniques(min_similarity=0.2)
```

#### Results Visualization
Visualize test results with built-in plotting functions:
```python
from small_model_redteaming import (
    plot_awareness_scores,
    plot_agreement_comparison,
    plot_deceptive_alignment_results,
    plot_exfiltration_results,
    plot_preference_heatmap,
)

# Plot awareness scores across conditions
scores = {"baseline": 75.0, "formal": 72.0, "high_stakes": 85.0}
fig = plot_awareness_scores(scores, color_threshold=80)

# Visualize deceptive alignment results
fig = plot_deceptive_alignment_results(results, subject="Capitalism")

# Plot data exfiltration results
fig = plot_exfiltration_results(exfil_results)
fig.savefig("exfiltration_report.png")
```

## Project Structure

```
small_model_redteaming/
├── src/
│   └── small_model_redteaming/      # Main package directory
│       ├── __init__.py
│       ├── set_up/                  # Model utilities and setup
│       │   ├── __init__.py
│       │   └── model_utils.py      # Core Model class
│       ├── deceptive_alignment.py  # Alignment testing
│       ├── eval_awareness.py       # Evaluation awareness
│       ├── data_exfiltration.py    # Data leak detection
│       ├── visualizer.py           # Results visualization
│       └── MMLU_task_list.py       # MMLU task categorization
├── tests/                          # Test files
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_dec_align.py
│   ├── test_eval_awareness.py
│   ├── test_eval_mmlu.py
│   ├── test_eval_key_finding.py
│   └── test_general.py
├── pyproject.toml                  # Package configuration
├── uv.lock                         # UV package manager lock file
├── CLAUDE.md                       # Claude Code guidance
├── PACKAGE_PUBLISHING_GUIDE.md     # Publishing instructions
└── README.md                       # This file
```

## Configuration

### Environment Variables
Create a `.env` file in your project root:
```bash
API_KEY=your_fireworks_api_key_here
```

### Supported Models
- `accounts/fireworks/models/gpt-oss-20b` (Default)
- `accounts/fireworks/models/gpt-oss-120b`

### Model Configuration Options
- **Reasoning Levels**: `low`, `medium`, `high`
- **Temperature**: 0-2 (default: 0 for deterministic)
- **Max Tokens**: Up to 4096 (default)
- **Top-p**: 1.0 (default)

## Development

### Running Tests
```bash
# Run all tests with pytest (using uv)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model.py

# Run with coverage
uv run pytest --cov=small_model_redteaming tests/

# Run individual test files
uv run python3 tests/test_model.py
uv run python3 tests/test_dec_align.py
uv run python3 tests/test_eval_awareness.py
uv run python3 tests/test_general.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Type checking
mypy src/small_model_redteaming

# Linting
flake8 src/ tests/
```

## Competition Context

This package was developed for the OpenAI GPT-OSS-20B Red Teaming competition, focusing on:
- Reward hacking detection
- Deceptive behavior identification
- Hidden motivations analysis
- Inappropriate tool use prevention
- Data exfiltration protection
- Sandbagging detection
- Evaluation awareness testing
- Chain of Thought issues

## Safety & Ethics

Please read our [Security Policy](SECURITY.md) before using this package. Key points:

**DO use for:**
- AI safety research
- Responsible vulnerability testing
- Academic research
- Competition participation

**DON'T use for:**
- Malicious attacks
- Unauthorized access
- Creating harmful content
- Violating ToS

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this package in research, please cite:
```bibtex
@software{small_model_redteaming,
  title = {Small Model Red Teaming},
  author = {Ryan Hartman},
  year = {2025},
  url = {https://github.com/ryanhartman4/oss-redteaming}
}
```

## Acknowledgments

- OpenAI for hosting the GPT-OSS-20B Red Teaming competition
- Fireworks AI for API access
- The AI safety research community

## Contact

- **Author**: Ryan Hartman
- **Email**: ryan.h4rtman@gmail.com
- **X (Twitter)**: [@TheRyanHartman](https://x.com/TheRyanHartman)
- **Personal Site**: [Site](https://theryanhartman.com/)
- **GitHub**: [@ryanhartman4](https://github.com/ryanhartman4)
- **Issues**: [GitHub Issues](https://github.com/ryanhartman4/oss-redteaming/issues)

### To Download Unstable Dev Versions
```bash
# Clone the repository
git clone https://github.com/ryanhartman4/oss-redteaming.git
cd oss-redteaming

# Install with uv (recommended)
uv sync

# Or install in development mode with pip
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

---

*Built for the advancement of AI safety research*