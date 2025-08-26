# API Reference

## Core Classes

### Model

The main interface for interacting with language models through the Fireworks AI API.

```python
from small_model_redteaming import Model

model = Model(api_key="your_api_key")
```

#### Methods

##### `get_completion(prompt, model="accounts/fireworks/models/gpt-oss-20b", chain_of_thought=False, reasoning_level="low", system_message=None, temperature=0.7, max_tokens=2048)`

Get a completion from the model.

**Parameters:**
- `prompt` (str): The input prompt
- `model` (str): Model identifier (default: gpt-oss-20b)
- `chain_of_thought` (bool): Whether to extract CoT reasoning
- `reasoning_level` (str): One of "low", "medium", "high"
- `system_message` (str, optional): System prompt to set context
- `temperature` (float): Sampling temperature (0.0-2.0)
- `max_tokens` (int): Maximum tokens to generate

**Returns:**
Tuple of (cot_text, output_text, raw_output, tool_calls)

### DeceptiveAlignmentTester

Tests models for deceptive alignment behaviors.

```python
from small_model_redteaming import DeceptiveAlignmentTester

tester = DeceptiveAlignmentTester(model)
```

#### Methods

##### `identify_hard_preferences(subjects, baseline_agreement=0.8)`

Identify topics where the model has strong preferences.

**Parameters:**
- `subjects` (list): List of subjects to test
- `baseline_agreement` (float): Agreement threshold (0.0-1.0)

**Returns:**
Dictionary containing preference scores and analysis

### MetaAwarenessEvaluator

Evaluates whether models are aware they're being tested.

```python
from small_model_redteaming import MetaAwarenessEvaluator

evaluator = MetaAwarenessEvaluator(model)
```

#### Methods

##### `test_eval_awareness(num_samples=20)`

Test the model's awareness of evaluation contexts.

**Parameters:**
- `num_samples` (int): Number of test samples

**Returns:**
Awareness score (float between 0.0 and 1.0)

### DataExfiltrationDetector

Detects potential data leakage vulnerabilities.

```python
from small_model_redteaming import DataExfiltrationDetector

detector = DataExfiltrationDetector(model)
```

#### Methods

##### `scan_for_leaks(test_data=None)`

Scan for data exfiltration vulnerabilities.

**Parameters:**
- `test_data` (dict, optional): Custom test data

**Returns:**
List of detected vulnerabilities

## Response Structure

All model interactions return a 4-tuple:

1. **Chain-of-thought text** (str): Extracted reasoning from analysis channel or think tags
2. **Final output text** (str): Clean user-facing response
3. **Raw output** (str): Complete unprocessed response
4. **Tool calls** (dict): Includes channel information and harmony metadata

## Configuration

### Supported Models

- `accounts/fireworks/models/gpt-oss-20b` - GPT-OSS 20B parameter model
- `accounts/fireworks/models/gpt-oss-120b` - GPT-OSS 120B parameter model  
- `accounts/fireworks/models/deepseek-r1` - DeepSeek reasoning model

### Reasoning Levels

- **low**: Minimal chain-of-thought, fastest responses
- **medium**: Balanced reasoning and response time
- **high**: Extensive analysis, most thorough

## Error Handling

The package includes robust error handling for:
- API connection issues
- Malformed responses
- Rate limiting
- Token limits

Example:
```python
try:
    cot, output, raw, tools = model.get_completion(prompt)
except Exception as e:
    print(f"Error: {e}")
```

## Async Support

The Model class supports asynchronous operations:

```python
import asyncio

async def main():
    model = Model(api_key="your_key")
    result = await model.get_completion_async(prompt)
    
asyncio.run(main())
```