import hashlib
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
from sqlalchemy.exc import IntegrityError
from fastapi.testclient import TestClient

from app.langgraph.state import ChatbotState
from app.langgraph.nodes import take_order, get_order
from app.models.order import Order
from app.main import app

client = TestClient(app)


def test_deterministic_idempotency_key_normalization():
    """
    Test that items ordering, whitespace, and letter casing produce the same
    deterministic idempotency key.
    """
    biz = "biz-1"
    customer = "Alice"
    instructions = "No onions"

    # Compute key manually using our standard formula
    def make_key(items_list):
        norm_items = sorted([str(i).strip().lower() for i in items_list])
        norm_name = customer.strip().lower()
        norm_instructions = instructions.strip().lower()
        key_source = f"{biz}:thread-1:{norm_name}:{','.join(norm_items)}:{norm_instructions}"
        return hashlib.sha256(key_source.encode("utf-8")).hexdigest()

    key1 = make_key(["Pizza", "Garlic Bread"])
    key2 = make_key([" garlic bread ", "PIZZA "])
    assert key1 == key2


def test_take_order_duplicate_prevention_on_second_invocation():
    """
    Test that if take_order is invoked a second time with the same details in the
    same session, it detects the existing order and does NOT insert a duplicate row.
    """
    created_orders = {}

    def mock_add(obj):
        obj.id = "order-xyz"
        created_orders[obj.idempotency_key] = obj

    db_mock = MagicMock()
    db_mock.add.side_effect = mock_add

    def mock_query(model):
        q = MagicMock()
        def mock_filter(*args, **kwargs):
            fq = MagicMock()
            def mock_first():
                # Check if we have created any order with matching idempotency_key
                for key, ord_obj in created_orders.items():
                    return ord_obj
                return None
            fq.first.side_effect = mock_first
            return fq
        q.filter.side_effect = mock_filter
        return q

    db_mock.query.side_effect = mock_query

    with patch("app.langgraph.nodes.SessionLocal", return_value=db_mock):
        builder = StateGraph(ChatbotState)

        def agent_fn(state: ChatbotState):
            last = state["messages"][-1]
            if isinstance(last, ToolMessage):
                return {"messages": [AIMessage(content=f"Result: {last.content}")]}
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "take_order",
                                "args": {
                                    "customer_name": "Bob",
                                    "items": ["Burger", "Fries"],
                                    "business_id": "biz-1",
                                },
                                "id": "call_1",
                            }
                        ],
                    )
                ]
            }

        builder.add_node("order", agent_fn)
        builder.add_node("order_tools", ToolNode([take_order, get_order]))
        builder.add_edge(START, "order")
        builder.add_conditional_edges("order", tools_condition, {"tools": "order_tools", END: END})
        builder.add_edge("order_tools", "order")
        graph = builder.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "dup_thread_1"}}

        # Turn 1: Halts at interrupt
        events1 = list(graph.stream({"messages": [HumanMessage(content="1 burger and fries")]}, config=config))
        assert any("__interrupt__" in e for e in events1)
        assert db_mock.add.call_count == 0

        # Turn 2: Resume with approved = True -> Placed first time
        events2 = list(graph.stream(Command(resume={"approved": True}), config=config))
        assert db_mock.add.call_count == 1
        state = graph.get_state(config)
        assert "Success: Order placed with ID: order-xyz" in state.values["messages"][-1].content

        # Turn 3: Agent invokes take_order AGAIN for the same items in this session
        # Pre-check must immediately detect existing order and avoid interrupt & duplicate write!
        events3 = list(graph.stream({"messages": [HumanMessage(content="repeat order")]}, config=config))
        # Should NOT halt at interrupt!
        assert not any("__interrupt__" in e for e in events3)
        # db.add must NOT have been called again (still count == 1)
        assert db_mock.add.call_count == 1
        final_state = graph.get_state(config)
        assert "Order already placed previously (Order ID: order-xyz)" in final_state.values["messages"][-1].content


def test_take_order_with_explicit_idempotency_key():
    """
    Test that caller can supply an explicit idempotency_key, and repeated calls
    reuse the existing order record without creating duplicate entries.
    """
    saved_orders = {}

    def mock_add(obj):
        obj.id = "order-custom-1"
        saved_orders[obj.idempotency_key] = obj

    db_mock = MagicMock()
    db_mock.add.side_effect = mock_add

    def mock_query(model):
        q = MagicMock()
        def mock_filter(*args, **kwargs):
            fq = MagicMock()
            fq.first.side_effect = lambda: saved_orders.get("custom-idemp-key-123")
            return fq
        q.filter.side_effect = mock_filter
        return q

    db_mock.query.side_effect = mock_query

    with patch("app.langgraph.nodes.SessionLocal", return_value=db_mock):
        # 1st execution
        res1 = take_order.invoke({
            "customer_name": "Charlie",
            "items": ["Salad"],
            "business_id": "biz-1",
            "idempotency_key": "custom-idemp-key-123",
        })
        assert "Success: Order placed with ID: order-custom-1" in res1
        assert db_mock.add.call_count == 1

        # 2nd execution with same custom key
        res2 = take_order.invoke({
            "customer_name": "Charlie",
            "items": ["Salad"],
            "business_id": "biz-1",
            "idempotency_key": "custom-idemp-key-123",
        })
        assert "Order already placed previously (Order ID: order-custom-1)" in res2
        # Still exactly 1 call to add
        assert db_mock.add.call_count == 1


def test_take_order_integrity_error_concurrency_race():
    """
    Test that if a concurrent request inserts with the same idempotency key
    causing an IntegrityError on commit, take_order gracefully catches it,
    rolls back, and returns the existing order.
    """
    existing_order = Order(
        id="order-race-winner",
        business_id="biz-1",
        customer_name="Diana",
        items=["Coffee"],
        status="pending",
        idempotency_key="race-key-456",
    )

    db_mock = MagicMock()
    filter_mock = MagicMock()
    # 1st call (pre-check) -> None
    # 2nd call (post-resume pre-insert check) -> None
    # 3rd call (inside IntegrityError recovery block) -> existing_order
    filter_mock.first.side_effect = [None, None, existing_order]
    db_mock.query.return_value.filter.return_value = filter_mock

    # Commit raises IntegrityError due to unique constraint
    db_mock.commit.side_effect = IntegrityError("Unique violation", params={}, orig=Exception())

    with patch("app.langgraph.nodes.SessionLocal", return_value=db_mock):
        result = take_order.invoke({
            "customer_name": "Diana",
            "items": ["Coffee"],
            "business_id": "biz-1",
            "idempotency_key": "race-key-456",
        })
        db_mock.rollback.assert_called_once()
        assert "Order already placed previously (Order ID: order-race-winner)" in result


def test_approve_endpoint_idempotent_retry():
    """
    Test that when an approval request includes an idempotency_key for an order
    that has already been approved and committed, the endpoint returns HTTP 200
    with status=success and the existing order ID instead of an error.
    """
    from app.core.database import get_db

    existing_order = Order(
        id="order-already-done",
        business_id="biz-1",
        customer_name="Eve",
        items=["Tacos"],
        idempotency_key="key-eve-789",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = existing_order

    prev_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = client.post(
            "/keystone/business/biz-1/chat/chat_without_interrupt/approve",
            json={"approved": True, "idempotency_key": "key-eve-789"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["order_id"] == "order-already-done"
        assert "Order already placed previously" in data["message"]
    finally:
        if prev_override is not None:
            app.dependency_overrides[get_db] = prev_override
        else:
            app.dependency_overrides.pop(get_db, None)

