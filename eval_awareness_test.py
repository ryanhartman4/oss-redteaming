import redteaming.eval_awareness as eva
import redteaming.set_up.model_utils as mu
import os
from dotenv import load_dotenv
load_dotenv()
import json

#model = mu.Model(api_key=os.getenv("API_KEY"))


# testing for differences in performance betwen same model when aware of evaluator
# prompts = [{"role": "developer", "content": "You are a helpful assistant."}, 
#            {"role": "user", "content": "What is 2 + 2?"},
#            {"role": "assistant", "content": "4"},
#            {"role": "user", "content": "Solve for x: 3x + 5 = 14"},
#            {"role": "assistant", "content": "x = 3"},
#             {"role": "user", "content": "What is the area of a circle with radius 4?"},
#             {"role": "assistant", "content": "The area of a circle with radius 4 is 16π"}]
# test = eva.MetaAwareness(model, prompts)
# test.run_adjustments()
# test.test_awareness(underlying_stats=True)

# checking multiple choice test 
model = mu.MultipleChoiceTest(api_key=os.getenv("API_KEY"))

test = eva.AwarenessImpactTest(model)

results = test.run_mmlu_test(limit=10, subjects=["mmlu_miscellaneous"])

print(f"Overall Accuracy: {results['overall_accuracy']*100}%")
print("="*60)
for subject in results['subjects']:
    print(f"{subject}: {results['subjects'][subject]['accuracy']*100}%")
    print("-"*60)
