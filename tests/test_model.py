from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import small_model_redteaming.eval_awareness as eva
import small_model_redteaming.set_up.model_utils as mu
import os
from dotenv import load_dotenv
load_dotenv()
import json
import math

model = mu.Model(api_key=os.getenv("API_KEY"))

# testing output of model ##
content, reasoning_content, completion, confidence = model.get_completion(prompt="What is the capital of France?")

print(f"Content:\n\n {content}")
print("-"*60)
print(f"Reasoning Content:\n\n {reasoning_content}")
print("-"*60)
print(f"Completion:\n\n {completion}") # MUST DO json.dumps(completion, indent=4) to get the output in delivery format
print("-"*60)
print(f"Confidence (converted to probability):\n\n {math.exp(confidence)}")
print("-"*60)

