# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-08-25

### Added
- Initial release of small_model_redteaming package
- Core red teaming modules:
  - Deceptive alignment detection using log probability analysis
  - Evaluation awareness testing with gradient prompts and meta-awareness tests
  - Data exfiltration vulnerability assessment
  - Results visualization and analysis tools
- Model interface layer supporting:
  - Chain-of-Thought (CoT) extraction (legacy `<think>` tags and Harmony format)
  - Tool call tracking and categorization
  - Synchronous, asynchronous, and streaming API modes
  - Robust response parsing for various output formats
- Integration with Fireworks AI API for GPT-OSS-20B model
- PreGenAnalyzer for lm_eval interface compatibility
- MMLU task integration with difficulty categorization
- Comprehensive test suite
- Documentation and usage examples
- Security policy and responsible use guidelines

### Changed
- Renamed package from `redteaming` to `small_model_redteaming` for clarity
- Restructured project to follow Python packaging best practices
- Moved source code to `src/` directory structure
- Updated test files to follow pytest naming conventions

### Security
- Added SECURITY.md with responsible use guidelines
- Implemented safety considerations for AI vulnerability research

[Unreleased]: https://github.com/ryanhartman4/small-model-redteaming/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ryanhartman4/small-model-redteaming/releases/tag/v0.1.0