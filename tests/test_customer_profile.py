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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.langgraph.state import ChatbotState
from app.langgraph.nodes import get_customer_profile, take_order, get_order
from app.models.customer import Customer
from app.main import seed_demo_customers


def test_seed_demo_customers():
    """Test that seed_demo_customers adds demo profiles into DB at startup."""
    mock_db = MagicMock()
    # Simulate DB where customer does not exist yet
    mock_db.query().filter().first.return_value = None

    with patch("app.main.SessionLocal", return_value=mock_db):
        seed_demo_customers()
        # Verify db.add was called for demo customers (Vivek and Alice)
        assert mock_db.add.call_count == 2
        mock_db.commit.assert_called_once()


def test_get_customer_profile_from_db():
    """Test retrieving customer profile from the database."""
    mock_db = MagicMock()
    mock_customer = Customer(
        id="cust-1",
        name="Sarah Connor",
        phone_number="9998887777",
        delivery_address="101 Tech Way",
        preferences="Gluten-free",
        frequent_items=["Salad", "Iced Tea"],
    )
    mock_db.query().filter().first.return_value = mock_customer

    with patch("app.langgraph.nodes.SessionLocal", return_value=mock_db):
        res = get_customer_profile.invoke({"phone_number": "9998887777"})
        assert "Sarah Connor" in res
        assert "101 Tech Way" in res
        assert "Gluten-free" in res
        assert "Salad, Iced Tea" in res


def test_get_customer_profile_not_found():
    """Test querying a phone number that does not exist in the database."""
    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = None

    with patch("app.langgraph.nodes.SessionLocal", return_value=mock_db):
        unknown_res = get_customer_profile.invoke({"phone_number": "5555555555"})
        assert "No customer profile found" in unknown_res


def test_autonomous_profile_lookup_and_order_loop():
    """
    Test the multi-step autonomous tool loop:
    1. Agent notices customer name is missing, calls get_customer_profile.
    2. ToolNode executes get_customer_profile and returns ToolMessage.
    3. Loop routes back to Order Agent.
    4. Order Agent reads customer name from profile and creates the order summary.
    """
    mock_db = MagicMock()
    mock_customer = Customer(
        id="cust-2",
        name="Alex Smith",
        phone_number="9876543210",
        delivery_address="123 Oak St",
        preferences="Mild spice",
        frequent_items=["Pad Thai"],
    )
    mock_db.query().filter().first.return_value = mock_customer

    with patch("app.langgraph.nodes.SessionLocal", return_value=mock_db):
        step = 0

        def mock_order_agent(state: ChatbotState):
            nonlocal step
            step += 1
            last_msg = state["messages"][-1]

            # Step 1: Missing name -> agent autonomously calls get_customer_profile
            if step == 1:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "get_customer_profile",
                                    "args": {"phone_number": state.get("phone_number")},
                                    "id": "lookup_call_1",
                                }
                            ],
                        )
                    ]
                }
            # Step 2: Agent receives ToolMessage with Alex Smith profile -> summarizes order
            elif isinstance(last_msg, ToolMessage):
                return {
                    "messages": [
                        AIMessage(
                            content="Hello Alex! I see your address is 123 Oak St. "
                            "Would you like me to confirm your order of Pad Thai?"
                        )
                    ]
                }
            return {"messages": [AIMessage(content="Order finished.")]}

        builder = StateGraph(ChatbotState)
        builder.add_node("order", mock_order_agent)
        builder.add_node("order_tools", ToolNode([get_customer_profile, take_order, get_order]))
        builder.add_edge(START, "order")
        builder.add_conditional_edges(
            "order", tools_condition, {"tools": "order_tools", END: END}
        )
        builder.add_edge("order_tools", "order")
        graph = builder.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "test_chaining_thread"}}

        events = list(
            graph.stream(
                {
                    "messages": [HumanMessage(content="Order my usual")],
                    "phone_number": "9876543210",
                    "business_id": "biz-1",
                },
                config=config,
            )
        )

        # Verify that ToolNode executed get_customer_profile in the stream
        tool_events = [e for e in events if "order_tools" in e]
        assert len(tool_events) == 1
        tool_msg = tool_events[0]["order_tools"]["messages"][0]
        assert tool_msg.name == "get_customer_profile"
        assert "Alex Smith" in tool_msg.content

        # Verify that the graph looped back to the agent and produced the personalized summary
        final_state = graph.get_state(config)
        final_response = final_state.values["messages"][-1].content
        assert "Hello Alex!" in final_response
        assert "123 Oak St" in final_response
        assert "Pad Thai" in final_response
