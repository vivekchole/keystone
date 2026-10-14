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
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.langgraph.state import ChatbotState
from app.langgraph.nodes import take_order, get_order
from app.models.business import Business
from app.models.order import Order
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import get_db

client = TestClient(app)


def test_take_order_interrupt_and_resume():
    """
    Test that take_order halts at interrupt() with the order details,
    and only executes the DB write when resumed with approved=True.
    """
    db_mock = MagicMock()
    order_instance = Order(id="order-123", customer_name="Alice", items=["Pizza"])
    db_mock.query().filter().first.return_value = order_instance
    db_mock.add.side_effect = lambda obj: setattr(obj, "id", "order-123")

    with patch("app.langgraph.nodes.SessionLocal", return_value=db_mock):
        def mock_order_agent(state: ChatbotState):
            last = state["messages"][-1]
            if isinstance(last, ToolMessage):
                return {"messages": [AIMessage(content=f"Agent confirmed: {last.content}")]}
            return {
                "messages": [
                    AIMessage(
                        content="Placing order now",
                        tool_calls=[
                            {
                                "name": "take_order",
                                "args": {
                                    "customer_name": "Alice",
                                    "items": ["Pizza"],
                                    "business_id": "biz-1",
                                },
                                "id": "call_1",
                            }
                        ],
                    )
                ]
            }

        builder = StateGraph(ChatbotState)
        builder.add_node("order", mock_order_agent)
        builder.add_node("order_tools", ToolNode([take_order, get_order]))
        builder.add_edge(START, "order")
        builder.add_conditional_edges(
            "order", tools_condition, {"tools": "order_tools", END: END}
        )
        builder.add_edge("order_tools", "order")
        test_graph = builder.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "approval_test_1"}}

        # Turn 1: Invocation triggers the interrupt before database write
        events = list(
            test_graph.stream(
                {
                    "messages": [HumanMessage(content="I want 1 pizza")],
                    "business_id": "biz-1",
                },
                config=config,
            )
        )

        # Ensure interrupt was emitted
        interrupt_events = [e for e in events if "__interrupt__" in e]
        assert len(interrupt_events) == 1
        interrupt_payload = interrupt_events[0]["__interrupt__"][0].value
        assert interrupt_payload["action"] == "order_approval"
        assert interrupt_payload["customer_name"] == "Alice"
        assert interrupt_payload["items"] == ["Pizza"]

        # Ensure DB add was NOT called yet
        db_mock.add.assert_not_called()

        # Turn 2: Resume with approved = True
        resume_events = list(
            test_graph.stream(
                Command(resume={"approved": True}),
                config=config,
            )
        )

        # Database write MUST have been called upon approval
        db_mock.add.assert_called_once()
        db_mock.commit.assert_called_once()

        # Final state check
        final_state = test_graph.get_state(config)
        final_msg = final_state.values["messages"][-1].content
        assert "Agent confirmed: Success: Order placed with ID: order-123" in final_msg


def test_take_order_interrupt_rejection():
    """
    Test that when an order approval is rejected, the database write
    is NOT executed and cancellation is reported.
    """
    db_mock = MagicMock()
    with patch("app.langgraph.nodes.SessionLocal", return_value=db_mock):
        def mock_order_agent(state: ChatbotState):
            last = state["messages"][-1]
            if isinstance(last, ToolMessage):
                return {"messages": [AIMessage(content=f"Agent: {last.content}")]}
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "take_order",
                                "args": {
                                    "customer_name": "Bob",
                                    "items": ["Burger"],
                                    "business_id": "biz-1",
                                },
                                "id": "call_2",
                            }
                        ],
                    )
                ]
            }

        builder = StateGraph(ChatbotState)
        builder.add_node("order", mock_order_agent)
        builder.add_node("order_tools", ToolNode([take_order, get_order]))
        builder.add_edge(START, "order")
        builder.add_conditional_edges(
            "order", tools_condition, {"tools": "order_tools", END: END}
        )
        builder.add_edge("order_tools", "order")
        test_graph = builder.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "approval_test_2"}}

        # Turn 1: Triggers interrupt
        list(
            test_graph.stream(
                {
                    "messages": [HumanMessage(content="Order burger")],
                    "business_id": "biz-1",
                },
                config=config,
            )
        )

        # Turn 2: Resume with approved = False
        resume_events = list(
            test_graph.stream(
                Command(resume={"approved": False, "reason": "User changed mind"}),
                config=config,
            )
        )

        # Database add must NOT be called
        db_mock.add.assert_not_called()

        final_state = test_graph.get_state(config)
        final_msg = final_state.values["messages"][-1].content
        assert "Order not placed: User changed mind" in final_msg


def test_approve_endpoint_no_pending_interrupt():
    """Test that POST /approve returns an error if no order is pending approval."""
    response = client.post(
        "/keystone/business/biz-123/chat/non_existent_chat/approve",
        json={"approved": True},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert "No order pending approval" in response.json()["message"]
