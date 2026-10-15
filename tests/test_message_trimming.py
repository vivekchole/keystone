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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from app.langgraph.nodes import (
    trim_conversation_messages,
    query_node,
    order_node,
    feedback_node,
    complaint_node,
)
from app.langgraph.state import ChatbotState


def test_trim_under_limit():
    msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
    result = trim_conversation_messages(msgs, max_messages=10)
    assert len(result) == 2
    assert result == msgs


def test_trim_empty():
    assert trim_conversation_messages([]) == []


def test_trim_over_limit_starts_on_human():
    """
    Ensure trimming keeps the most recent messages up to max_messages
    and guarantees the slice starts on a HumanMessage.
    """
    msgs = [
        HumanMessage(content="m1"),
        AIMessage(content="m2"),
        HumanMessage(content="m3"),
        AIMessage(content="m4"),
        HumanMessage(content="m5"),
        AIMessage(content="m6"),
        HumanMessage(content="m7"),
        AIMessage(content="m8"),
    ]

    result = trim_conversation_messages(msgs, max_messages=4)
    assert len(result) == 4
    assert result[0].content == "m5"
    assert result[-1].content == "m8"
    assert isinstance(result[0], HumanMessage)


def test_trim_preserves_valid_tool_sequence():
    """
    Ensure trimming never starts on an orphaned ToolMessage.
    """
    msgs = [
        HumanMessage(content="old user message"),
        AIMessage(content="old assistant message"),
        HumanMessage(content="order 2 coffees"),
        AIMessage(content="", tool_calls=[{"name": "take_order", "args": {}, "id": "call_1"}]),
        ToolMessage(content="order placed", tool_call_id="call_1"),
        AIMessage(content="Your order is placed!"),
        HumanMessage(content="can you check status?"),
        AIMessage(content="Checking status now."),
    ]

    # If we request 5 messages, a naive slice [-5:] would start on ToolMessage!
    # With start_on='human', it must safely slide forward to 'can you check status?'
    result = trim_conversation_messages(msgs, max_messages=5)
    assert isinstance(result[0], HumanMessage)
    assert not isinstance(result[0], ToolMessage)


def test_node_invocations_use_trimmed_context():
    """
    Verify that query_node passes only trimmed messages to the LLM
    even when state contains many historical messages.
    """
    # Create 30 historical messages
    history = []
    for i in range(15):
        history.append(HumanMessage(content=f"User turn {i}"))
        history.append(AIMessage(content=f"Bot reply {i}"))

    state: ChatbotState = {
        "messages": history,
        "business_context": "Test biz context",
        "business_id": "test-biz",
    }

    captured_messages = []

    mock_llm = MagicMock()

    def mock_invoke(messages_arg, **kwargs):
        captured_messages.extend(messages_arg)
        return AIMessage(content="Mocked response")

    mock_llm.invoke.side_effect = mock_invoke

    with patch("app.langgraph.nodes.llm", mock_llm), \
         patch("app.langgraph.nodes.MAX_CONTEXT_MESSAGES", 6):
        res = query_node(state)
        assert len(res["messages"]) == 1

        # Captured messages should have 1 SystemMessage + at most 6 trimmed messages
        assert isinstance(captured_messages[0], SystemMessage)
        assert len(captured_messages) <= 7  # 1 sys + 6 max trimmed
        # Original state list must remain intact (30 items)
        assert len(state["messages"]) == 30


def test_order_node_uses_trimmed_context():
    """
    Verify order_node passes trimmed messages to order_llm.
    """
    history = []
    for i in range(15):
        history.append(HumanMessage(content=f"Order user turn {i}"))
        history.append(AIMessage(content=f"Order bot reply {i}"))

    state: ChatbotState = {
        "messages": history,
        "business_context": "Order biz context",
        "business_id": "test-biz",
        "phone_number": "9876543210",
    }

    captured_messages = []
    mock_order_llm = MagicMock()

    def mock_order_invoke(messages_arg, **kwargs):
        captured_messages.extend(messages_arg)
        return AIMessage(content="Mocked order response")

    mock_order_llm.invoke.side_effect = mock_order_invoke

    with patch("app.langgraph.nodes.order_llm", mock_order_llm), \
         patch("app.langgraph.nodes.MAX_CONTEXT_MESSAGES", 6):
        res = order_node(state)
        assert len(res["messages"]) == 1
        assert isinstance(captured_messages[0], SystemMessage)
        assert len(captured_messages) <= 7
        assert len(state["messages"]) == 30
