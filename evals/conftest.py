import os
import pytest
import dotenv
from deepeval.models import GeminiModel

dotenv.load_dotenv()


@pytest.fixture(scope="session")
def gemini_judge():
    """Provides a shared DeepEval Gemini judge model instance."""
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash-lite")
    api_key = os.getenv("LLM_MODEL_KEY")
    if not api_key:
        pytest.skip("LLM_MODEL_KEY not set in environment. Skipping LLM evals.")
    return GeminiModel(model=model_name, api_key=api_key)
