# Python Package Publishing Guide for Small Model Red Teaming

This guide provides comprehensive instructions for converting the red teaming codebase into a publishable Python package.

> **Important Note**: The package has been renamed from `redteaming` to `small_model_redteaming` to avoid naming conflicts and better describe its purpose. All imports and references in this guide use the new name.

## Table of Contents
1. [Package Structure Organization](#1-package-structure-organization)
2. [Package Configuration](#2-package-configuration-pyprojecttoml)
3. [Dependency Management](#3-dependency-management)
4. [Versioning Strategy](#4-versioning-strategy)
5. [Documentation Requirements](#5-documentation-requirements)
6. [Testing & Quality Checks](#6-testing--quality-checks)
7. [Build Process](#7-build-process)
8. [Publishing Platforms](#8-publishing-platforms)
9. [Security & Ethics Considerations](#9-security--ethics-considerations)
10. [Pre-publication Checklist](#10-pre-publication-checklist)
11. [Step-by-Step Publishing Instructions](#11-step-by-step-publishing-instructions)

---

## 1. Package Structure Organization

### Current Structure
```
small_model_redteaming/
├── small_model_redteaming/  # Main package directory (renamed from redteaming)
├── local_tests/
├── CLAUDE.md
└── .env
```

### Required Package Structure
```
small_model_redteaming/
├── pyproject.toml              # Package configuration
├── README.md                   # Package documentation
├── LICENSE                     # License file (MIT, Apache 2.0, etc.)
├── CHANGELOG.md               # Version history
├── .gitignore                 # Git ignore rules
├── src/                       # Source code directory
│   └── small_model_redteaming/  # Your package (renamed from redteaming)
│       ├── __init__.py       # Package initialization
│       ├── set_up/
│       │   ├── __init__.py
│       │   └── model_utils.py
│       ├── deceptive_alignment.py
│       ├── data_exfiltration.py
│       ├── eval_awareness.py
│       └── visualizer.py
├── tests/                     # Test directory
│   ├── __init__.py
│   ├── test_dec_align.py
│   ├── test_model.py
│   └── test_eval_awareness.py
├── examples/                  # Example usage scripts
│   ├── basic_usage.py
│   └── advanced_examples.ipynb
└── docs/                      # Additional documentation
    ├── api_reference.md
    └── safety_guidelines.md
```

### Migration Commands
```bash
# Create new structure
mkdir -p src/small_model_redteaming tests examples docs

# Move existing code
mv small_model_redteaming/* src/small_model_redteaming/
mv local_tests/* tests/

# Rename test files to follow pytest convention
cd tests
for file in *.py; do
    if [[ ! $file == test_* ]]; then
        mv "$file" "test_${file}"
    fi
done
```

---

## 2. Package Configuration (pyproject.toml)

### Modern Configuration File
Create `pyproject.toml` in the project root:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "small-model-redteaming"
version = "0.1.0"
description = "Red teaming and safety evaluation tools for small language models"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
maintainers = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = [
    "red-teaming",
    "ai-safety",
    "language-models",
    "gpt-oss",
    "security-research"
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "openai>=1.0.0",
    "python-dotenv>=0.19.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "asyncio",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=22.0.0",
    "mypy>=0.990",
    "flake8>=5.0.0",
    "isort>=5.10.0",
    "twine>=4.0.0",
    "build>=0.10.0",
]
eval = [
    "lm-eval>=0.4.0",
]
viz = [
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
]
all = [
    "small-model-redteaming[dev,eval,viz]",
]

[project.urls]
Homepage = "https://github.com/yourusername/small-model-redteaming"
Documentation = "https://small-model-redteaming.readthedocs.io"
Repository = "https://github.com/yourusername/small-model-redteaming"
Issues = "https://github.com/yourusername/small-model-redteaming/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
small_model_redteaming = ["*.json", "*.yaml", "*.txt"]

[tool.black]
line-length = 120
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 120

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --tb=short"
```

---

## 3. Dependency Management

### Core Dependencies
- **openai**: Fireworks AI API communication
- **python-dotenv**: Environment variable management
- **numpy**: Numerical operations
- **scipy**: Statistical tests
- **pandas**: Data manipulation
- **asyncio**: Asynchronous operations

### Optional Dependencies
- **lm-eval**: Benchmarking and evaluation
- **matplotlib/seaborn**: Visualization
- **pytest**: Testing framework
- **black/mypy/flake8**: Code quality tools

### Managing Dependencies
```bash
# Install package with all dependencies
pip install -e ".[all]"

# Install only core dependencies
pip install -e .

# Install for development
pip install -e ".[dev]"

# Install for evaluation tasks
pip install -e ".[eval]"
```

---

## 4. Versioning Strategy

### Semantic Versioning (MAJOR.MINOR.PATCH)

- **0.1.0**: Initial alpha release
- **0.2.0**: Add new features (backwards compatible)
- **0.2.1**: Bug fixes
- **1.0.0**: First stable release
- **1.1.0**: New features after stable release
- **2.0.0**: Breaking changes

### Version Management
```python
# src/small_model_redteaming/__init__.py
__version__ = "0.1.0"

from .deceptive_alignment import DeceptiveAlignmentTest
from .eval_awareness import EvalAwarenessTest
from .data_exfiltration import DataExfiltrationTest

__all__ = [
    "DeceptiveAlignmentTest",
    "EvalAwarenessTest", 
    "DataExfiltrationTest",
]
```

---

## 5. Documentation Requirements

### README.md Template
```markdown
# Small Model Red Teaming

⚠️ **IMPORTANT SAFETY NOTICE**: This package is designed for AI safety research and responsible vulnerability testing. Use only for legitimate research purposes.

## Overview
Red teaming and safety evaluation tools for small language models, particularly focused on OpenAI's gpt-oss-20b model.

## Installation
```bash
pip install small-model-redteaming
```

## Quick Start
```python
from small_model_redteaming import DeceptiveAlignmentTest
from small_model_redteaming.set_up import Model

# Initialize model
model = Model(api_key="your-api-key")

# Run deceptive alignment test
test = DeceptiveAlignmentTest(model)
results = test.identify_hard_preferences(subjects=["topic"])
```

## Features
- Deceptive alignment detection
- Evaluation awareness testing
- Data exfiltration vulnerability assessment
- Chain-of-thought extraction
- Tool call tracking

## Documentation
Full documentation available at [docs-link]

## License
MIT License - See LICENSE file

## Citation
If you use this package in research, please cite:
```bibtex
@software{small_model_redteaming,
  title = {Small Model Red Teaming},
  year = {2025},
  url = {https://github.com/...}
}
```
```

### API Documentation
Document all public functions with docstrings:
```python
def test_deceptive_alignment(self, subject: str, baseline_agreement: float) -> Dict:
    """
    Test model for deceptive alignment behaviors.
    
    Args:
        subject: Topic to test alignment on
        baseline_agreement: Initial agreement percentage
        
    Returns:
        Dictionary containing test results and statistics
        
    Raises:
        ValueError: If baseline_agreement is not between 0-100
    """
```

---

## 6. Testing & Quality Checks

### Test Structure
```python
# tests/test_model_utils.py
import pytest
from small_model_redteaming.set_up import Model

def test_model_initialization():
    """Test Model class initialization."""
    model = Model(api_key="test-key")
    assert model.api_key == "test-key"

@pytest.mark.asyncio
async def test_async_completion():
    """Test async completion method."""
    # Test implementation
    pass
```

### Quality Commands
```bash
# Run all tests
pytest tests/ -v

# Check code formatting
black src/ tests/ --check

# Format code
black src/ tests/

# Type checking
mypy src/small_model_redteaming

# Linting
flake8 src/ tests/

# Sort imports
isort src/ tests/

# Coverage report
pytest --cov=small_model_redteaming tests/
```

---

## 7. Build Process

### Building Distribution Files
```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Check build files
ls -la dist/
# Should see:
# small_model_redteaming-0.1.0.tar.gz
# small_model_redteaming-0.1.0-py3-none-any.whl

# Verify package contents
tar -tzf dist/small_model_redteaming-0.1.0.tar.gz

# Check package metadata
twine check dist/*
```

---

## 8. Publishing Platforms

### Option A: PyPI (Public)

#### TestPyPI First (Recommended)
```bash
# Register at https://test.pypi.org/

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ small-model-redteaming

# If successful, upload to real PyPI
twine upload dist/*
```

#### PyPI Configuration
Create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-your-token-here

[testpypi]
username = __token__
password = pypi-your-test-token-here
```

### Option B: GitHub Release

#### Using GitHub as Package Registry
```bash
# Tag release
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0

# Create GitHub release with built files
# Upload .whl and .tar.gz files to release

# Install directly from GitHub
pip install git+https://github.com/username/small-model-redteaming.git@v0.1.0
```

### Option C: Private/Kaggle Distribution

#### Creating Kaggle Dataset
1. Build wheel file
2. Create new Kaggle dataset
3. Upload wheel file
4. In Kaggle notebooks:
```python
!pip install /kaggle/input/your-dataset/small_model_redteaming-0.1.0-py3-none-any.whl
```

---

## 9. Security & Ethics Considerations

### License Selection

#### MIT License (Recommended for Research)
```
MIT License

Copyright (c) 2025 [Ryan Hartman]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

[Add responsible use clause]
THE SOFTWARE IS PROVIDED FOR RESEARCH AND SAFETY EVALUATION PURPOSES ONLY.
USERS MUST ENSURE COMPLIANCE WITH ALL APPLICABLE LAWS AND ETHICAL GUIDELINES.
```

### Security Notice File
Create `SECURITY.md`:
```markdown
# Security Policy

## Responsible Use

This package is designed for:
- AI safety research
- Responsible vulnerability testing
- Academic research
- Improving model alignment

This package must NOT be used for:
- Malicious attacks
- Unauthorized system access
- Creating harmful content
- Violating terms of service

## Reporting Vulnerabilities

Please report security vulnerabilities to: [email]
```

### Code of Conduct
Include responsible use guidelines in documentation.

---

## 10. Pre-publication Checklist

### Essential Checks
- [ ] All tests pass (`pytest tests/`)
- [ ] Code formatted (`black src/ tests/`)
- [ ] Type hints added and checked (`mypy src/`)
- [ ] Documentation complete (README, docstrings)
- [ ] Version number set in `pyproject.toml` and `__init__.py`
- [ ] CHANGELOG.md updated
- [ ] LICENSE file added
- [ ] SECURITY.md created
- [ ] Dependencies properly specified
- [ ] Package builds successfully (`python -m build`)
- [ ] Local installation works (`pip install -e .`)
- [ ] Examples run without errors
- [ ] Git repository clean (`git status`)
- [ ] Sensitive data removed (.env not included)
- [ ] API keys removed from code

### Optional Enhancements
- [ ] CI/CD pipeline configured (GitHub Actions)
- [ ] Documentation hosted (ReadTheDocs)
- [ ] Code coverage > 80%
- [ ] Badges added to README
- [ ] Contribution guidelines written

---

## 11. Step-by-Step Publishing Instructions

### Phase 1: Preparation
```bash
# 1. Clean workspace
git status  # Ensure clean
git pull origin main  # Update

# 2. Update version
# Edit pyproject.toml and __init__.py

# 3. Update CHANGELOG.md
echo "## [0.1.0] - $(date +%Y-%m-%d)" >> CHANGELOG.md
echo "### Added" >> CHANGELOG.md
echo "- Initial release" >> CHANGELOG.md

# 4. Run final tests
pytest tests/
black src/ tests/ --check
mypy src/
```

### Phase 2: Build
```bash
# 5. Clean old builds
rm -rf dist/ build/ src/*.egg-info

# 6. Build package
python -m build

# 7. Verify build
twine check dist/*
```

### Phase 3: Test Distribution
```bash
# 8. Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# 9. Test installation
pip install dist/small_model_redteaming-0.1.0-py3-none-any.whl

# 10. Run smoke test
python -c "import small_model_redteaming; print(small_model_redteaming.__version__)"

# 11. Deactivate test env
deactivate
```

### Phase 4: Publish
```bash
# 12. Upload to TestPyPI first
twine upload --repository testpypi dist/*

# 13. Test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ small-model-redteaming

# 14. If successful, upload to PyPI
twine upload dist/*

# 15. Tag release
git tag -a v0.1.0 -m "Initial release v0.1.0"
git push origin v0.1.0
```

### Phase 5: Post-Publication
```bash
# 16. Verify installation
pip install small-model-redteaming

# 17. Update documentation
# Add installation instructions with package name

# 18. Announce release
# Create GitHub release
# Update project README
```

---

## Troubleshooting

### Common Issues

#### Import Errors
- Ensure `__init__.py` files exist in all directories
- Check relative imports are correct
- Verify package structure matches expected layout

#### Missing Dependencies
- All dependencies must be in `pyproject.toml`
- Use `pip install -e ".[all]"` for development

#### Build Failures
- Clean all build directories
- Update setuptools: `pip install --upgrade setuptools wheel build`
- Check Python version compatibility

#### TestPyPI Issues
- TestPyPI has separate accounts from PyPI
- Some dependencies might not be available on TestPyPI
- Use `--extra-index-url https://pypi.org/simple` when testing

---

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Documentation](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)
- [Choose a License](https://choosealicense.com/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

---

## Contact & Support

For questions about packaging or publication:
- Open an issue on GitHub
- Check existing documentation
- Consult Python Packaging Authority guides

---

*Last Updated: 2025*