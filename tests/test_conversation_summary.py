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
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage,
    SystemMessage,
)
from app.langgraph.nodes import (
    find_safe_cutoff,
    should_summarize,
    summarize_conversation_node,
    query_node,
    order_node,
    get_system_prompt,
)
from app.langgraph.state import ChatbotState
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import get_db
from app.models.business import Business

client = TestClient(app)

mock_business = Business(
    id="biz-summary-test",
    name="Summary Test Business",
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


def test_should_summarize_threshold():
    # Below threshold (e.g. 5 messages vs threshold of 10)
    state_short: ChatbotState = {"messages": [HumanMessage(content=f"msg {i}") for i in range(5)]}
    assert should_summarize(state_short) == "end"

    # At or above threshold (10 messages)
    state_long: ChatbotState = {"messages": [HumanMessage(content=f"msg {i}") for i in range(10)]}
    assert should_summarize(state_long) == "summarize_conversation"


def test_find_safe_cutoff_aligns_to_human_message():
    msgs = [
        HumanMessage(content="u1", id="1"),
        AIMessage(content="a1", id="2"),
        HumanMessage(content="u2", id="3"),
        AIMessage(content="order something", tool_calls=[{"name": "take_order", "args": {}, "id": "c1"}], id="4"),
        ToolMessage(content="placed", tool_call_id="c1", id="5"),
        AIMessage(content="confirmed", id="6"),
        HumanMessage(content="u3", id="7"),
        AIMessage(content="a3", id="8"),
    ]
    # Buffer size 3 would target len - 3 = 5 ('confirmed'), which is an AIMessage
    # Safe cutoff should advance to index 6 ('u3')
    cutoff = find_safe_cutoff(msgs, buffer_size=3)
    assert cutoff == 6
    assert isinstance(msgs[cutoff], HumanMessage)


def test_summarize_conversation_node_execution():
    msgs = [
        HumanMessage(content="Hi, my name is John Doe", id="m1"),
        AIMessage(content="Hello John!", id="m2"),
        HumanMessage(content="I want to order a Margherita pizza", id="m3"),
        AIMessage(content="Sure, placing order for Margherita", id="m4"),
        HumanMessage(content="Also deliver to 123 Main St", id="m5"),
        AIMessage(content="Address noted.", id="m6"),
        HumanMessage(content="What time will it arrive?", id="m7"),
        AIMessage(content="Around 30 minutes.", id="m8"),
    ]

    state: ChatbotState = {
        "messages": msgs,
        "summary": None,
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Customer John ordered Margherita pizza to 123 Main St.")

    with patch("app.langgraph.nodes.llm", mock_llm), \
         patch("app.langgraph.nodes.SUMMARY_BUFFER_SIZE", 2):
        result = summarize_conversation_node(state)

        assert "summary" in result
        assert result["summary"] == "Customer John ordered Margherita pizza to 123 Main St."
        assert "messages" in result
        # Check that RemoveMessage objects were generated for the messages up to cutoff
        assert all(isinstance(m, RemoveMessage) for m in result["messages"])
        assert len(result["messages"]) > 0


def test_summary_injected_into_query_and_order_prompts():
    state: ChatbotState = {
        "messages": [HumanMessage(content="What was my address again?")],
        "business_context": "Sample Pizza Shop",
        "summary": "Customer previously asked for delivery to 123 Main St.",
    }

    captured_prompt = []

    def mock_query_invoke(msgs, **kwargs):
        captured_prompt.extend(msgs)
        return AIMessage(content="Your delivery address is 123 Main St.")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_query_invoke

    with patch("app.langgraph.nodes.llm", mock_llm):
        query_node(state)
        sys_msg = captured_prompt[0]
        assert isinstance(sys_msg, SystemMessage)
        assert "Summary of earlier conversation:" in sys_msg.content
        assert "123 Main St" in sys_msg.content


def test_get_chat_history_includes_summary():
    """
    Test that GET /{business_id}/chat/{chat_id} includes the summary in its response.
    """
    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "summary": "Customer John ordered 2 pizzas earlier.",
        "messages": [HumanMessage(content="Where is my order?"), AIMessage(content="It is on the way.")],
    }
    mock_snapshot.tasks = []

    with patch("app.api.chat.graph.aget_state", return_value=mock_snapshot):
        response = client.get(f"/keystone/business/{mock_business.id}/chat/summary-chat-thread")
        assert response.status_code == 200
        data = response.json()
        assert data.get("summary") == "Customer John ordered 2 pizzas earlier."
        assert len(data.get("messages", [])) == 2


def test_get_system_prompt_helper():
    """
    Test the common get_system_prompt helper function for template formatting and summary injection.
    """
    state_without_summary: ChatbotState = {"messages": []}
    sys_msg, version = get_system_prompt("feedback", state_without_summary)
    assert isinstance(sys_msg, SystemMessage)
    assert "Summary of earlier conversation:" not in sys_msg.content

    state_with_summary: ChatbotState = {
        "messages": [],
        "summary": "Customer loves cheese burst pizza.",
    }
    sys_msg_summary, version2 = get_system_prompt(
        "query",
        state_with_summary,
        business_context="Pizzeria Open Daily",
    )
    assert "Pizzeria Open Daily" in sys_msg_summary.content
    assert "Summary of earlier conversation:" in sys_msg_summary.content
    assert "cheese burst pizza" in sys_msg_summary.content
    assert version2 == "v1.0.0"

