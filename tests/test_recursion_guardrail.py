import json
import os
import sys
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver

os.environ["LLM_MODEL_NAME"] = "gpt-4o-mini"
os.environ["LLM_MODEL_KEY"] = "mock-key"

mock_saver = MemorySaver()
sys.modules["langgraph.checkpoint.postgres.aio"] = MagicMock()
sys.modules["langgraph.checkpoint.postgres.aio"].AsyncPostgresSaver = MagicMock(
    return_value=mock_saver
)

import pytest
from fastapi.testclient import TestClient
from langgraph.errors import GraphRecursionError
from app.main import app
from app.core.database import get_db
from app.models.business import Business
from app.api.chat import RECURSION_LIMIT

client = TestClient(app)

mock_business = Business(
    id="biz-recursion-test",
    name="Recursion Test Business",
    services=["service1"],
    timings="9 to 5",
    description="A test business",
)


class MockSession:
    def query(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return mock_business

    def add(self, obj):
        if not getattr(obj, "id", None):
            obj.id = mock_business.id

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def close(self):
        pass


def override_get_db():
    yield MockSession()


@pytest.fixture(autouse=True)
def setup_teardown_db():
    prev_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db
    yield
    if prev_override is not None:
        app.dependency_overrides[get_db] = prev_override
    else:
        app.dependency_overrides.pop(get_db, None)


def test_chat_stream_recursion_guardrail():
    """
    Test that when graph.astream raises GraphRecursionError,
    the endpoint catches it and returns a graceful SSE guardrail event
    instead of crashing the HTTP stream.
    """
    async def mock_recursive_stream(*args, **kwargs):
        config = kwargs.get("config", {})
        assert "recursion_limit" in config
        assert config["recursion_limit"] == RECURSION_LIMIT
        raise GraphRecursionError("Recursion limit of 15 reached without hitting a stop condition.")
        yield  # make it an async generator

    with patch("app.api.chat.graph.astream", side_effect=mock_recursive_stream):
        response = client.post(
            f"/keystone/business/{mock_business.id}/chat/recursion-thread-1",
            json={"message": "Please do something that loops infinitely"},
        )
        assert response.status_code == 200

        lines = [line for line in response.text.strip().split("\n") if line.startswith("data: ")]
        assert len(lines) >= 2

        # First event should be the guardrail message
        first_payload = json.loads(lines[0].replace("data: ", ""))
        assert first_payload.get("node") == "guardrail"
        assert first_payload.get("error_type") == "recursion_limit"
        assert "too many steps" in first_payload.get("content", "").lower()

        # Last event must be clean [DONE]
        assert lines[-1] == "data: [DONE]"


def test_approval_stream_recursion_guardrail():
    """
    Test that when graph.astream raises GraphRecursionError during order approval,
    the approval endpoint catches it and returns a graceful SSE guardrail event.
    """
    # Mock state snapshot with a pending interrupt task
    mock_task = MagicMock()
    mock_task.interrupts = [MagicMock(value={"action": "order_approval"})]
    mock_snapshot = MagicMock(tasks=[mock_task])

    async def mock_recursive_stream(*args, **kwargs):
        config = kwargs.get("config", {})
        assert "recursion_limit" in config
        assert config["recursion_limit"] == RECURSION_LIMIT
        raise GraphRecursionError("Recursion limit reached in approval flow")
        yield

    with patch("app.api.chat.graph.aget_state", return_value=mock_snapshot), \
         patch("app.api.chat.graph.astream", side_effect=mock_recursive_stream):
        response = client.post(
            f"/keystone/business/{mock_business.id}/chat/recursion-thread-2/approve",
            json={"approved": True},
        )
        assert response.status_code == 200

        lines = [line for line in response.text.strip().split("\n") if line.startswith("data: ")]
        assert len(lines) >= 2

        first_payload = json.loads(lines[0].replace("data: ", ""))
        assert first_payload.get("node") == "guardrail"
        assert first_payload.get("error_type") == "recursion_limit"
        assert "exceeded the maximum allowed steps" in first_payload.get("content", "").lower()

        assert lines[-1] == "data: [DONE]"
