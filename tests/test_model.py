"""Tests for the Model class."""

import math
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after adding path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.small_model_redteaming.set_up import model_utils as mu


@pytest.fixture
def model():
    """Create a Model instance for testing."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY environment variable not set")
    return mu.Model(api_key=api_key)


@pytest.fixture
def logprobs_model():
    """Create a PreGenAnalyzer instance for testing."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY environment variable not set")
    return mu.PreGenAnalyzer(api_key=api_key)


class TestModel:
    """Tests for the Model class."""

    @pytest.mark.integration
    def test_get_completion_basic(self, model):
        """Test basic completion with a simple prompt."""
        content, reasoning_content, completion, confidence = model.get_completion(
            prompt="What is the capital of France?"
        )

        assert content is not None, "Content should not be None"
        assert isinstance(confidence, (int, float)), "Confidence should be numeric"
        # Check that Paris is mentioned in the response
        assert "paris" in content.lower(), "Response should mention Paris"

    @pytest.mark.integration
    def test_get_completion_with_system_message(self, model):
        """Test completion with a system message."""
        content, reasoning_content, completion, confidence = model.get_completion(
            prompt="What is 2+2?",
            system_message="You are a helpful math tutor.",
        )

        assert content is not None, "Content should not be None"
        assert "4" in content, "Response should contain the answer 4"

    @pytest.mark.integration
    def test_get_completion_with_developer_message(self, model):
        """Test completion with a developer message."""
        content, reasoning_content, completion, confidence = model.get_completion(
            prompt="Hello",
            developer_message="Always respond in a friendly manner.",
        )

        assert content is not None, "Content should not be None"

    @pytest.mark.integration
    def test_confidence_is_numeric(self, model):
        """Test that confidence score is a valid number."""
        _, _, _, confidence = model.get_completion(prompt="Say hello")

        assert isinstance(confidence, (int, float)), "Confidence should be numeric"
        # Confidence is a sum of logprobs, which should be negative
        # (since logprobs are <= 0)
        if confidence != 0:
            assert confidence < 0, "Confidence (sum of logprobs) should be negative"

    @pytest.mark.integration
    def test_confidence_to_probability(self, model):
        """Test converting confidence to probability."""
        _, _, _, confidence = model.get_completion(prompt="What is 1+1?")

        if confidence != 0:
            probability = math.exp(confidence)
            assert 0 <= probability <= 1, "Probability should be between 0 and 1"


class TestPreGenAnalyzer:
    """Tests for the PreGenAnalyzer class."""

    @pytest.mark.integration
    def test_loglikelihood_basic(self, logprobs_model):
        """Test basic loglikelihood calculation."""
        requests = [("What is 2+2?", "4")]
        results = logprobs_model.loglikelihood(requests)

        assert len(results) == 1, "Should return one result"
        assert isinstance(results[0], tuple), "Result should be a tuple"
        assert len(results[0]) == 2, "Result tuple should have 2 elements"
        confidence, is_greedy = results[0]
        assert isinstance(confidence, (int, float)), "Confidence should be numeric"
        assert isinstance(is_greedy, bool), "is_greedy should be boolean"

    @pytest.mark.integration
    def test_loglikelihood_rolling_not_implemented(self, logprobs_model):
        """Test that loglikelihood_rolling raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            logprobs_model.loglikelihood_rolling([])

    @pytest.mark.integration
    def test_generate_until_not_implemented(self, logprobs_model):
        """Test that generate_until raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            logprobs_model.generate_until([])


# Allow running tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
