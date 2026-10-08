import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage

import sys
from unittest.mock import patch, MagicMock, AsyncMock
from langgraph.checkpoint.memory import MemorySaver

mock_saver = MemorySaver()
sys.modules['langgraph.checkpoint.postgres.aio'] = MagicMock()
sys.modules['langgraph.checkpoint.postgres.aio'].AsyncPostgresSaver = MagicMock(return_value=mock_saver)

from app.main import app
from app.models.business import Business

client = TestClient(app)

# Dummy business data to mock DB responses
mock_business = Business(
    id="123e4567-e89b-12d3-a456-426614174000",
    name="Test Business",
    services=["service1", "service2"],
    timings="9 to 5",
    description="A test business"
)

# 1. Mock the DB Session
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

from app.core.database import get_db

def override_get_db():
    yield MockSession()

app.dependency_overrides[get_db] = override_get_db


# @patch("app.api.chat.graph.astream"): Mocks the LangGraph execution stream. 
# This bypasses the actual graph execution, LLM calls, and node routing. 
# It only tests that our FastAPI endpoint correctly formats and streams 
# whatever mock response we give it. Actual LLM routing is NOT tested here.
@patch("app.api.chat.graph.astream")

# @patch("app.api.chat.graph.aget_state"): Mocks the LangGraph checkpointer read.
# This prevents it from hitting the database, allowing us to return a 
# predefined chat history state instead.
@patch("app.api.chat.graph.aget_state")
def test_happy_flow(mock_aget_state, mock_astream):
    
    # ---------------------------------------------------------
    # 1. Create Business
    # ---------------------------------------------------------
    response = client.post("/keystone/business/", json={
        "name": "Test Business",
        "services": ["service1", "service2"],
        "timings": "9 to 5",
        "description": "A test business"
    })
    assert response.status_code == 200
    business_id = response.json()["id"]
    assert business_id == mock_business.id

    # ---------------------------------------------------------
    # 2. Get Business
    # ---------------------------------------------------------
    response = client.get(f"/keystone/business/{business_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Business"

    # ---------------------------------------------------------
    # 3 & 4. Chat API - Ask about business details
    # ---------------------------------------------------------
    # Mocking astream to yield a simple response
    async def mock_stream(*args, **kwargs):
        yield {"query": {"messages": [AIMessage(content="We offer service1 and service2")]}}

    mock_astream.side_effect = mock_stream
    
    chat_id = "test_chat_1"
    response = client.post(
        f"/keystone/business/{business_id}/chat/{chat_id}",
        json={"message": "What services do you offer?"}
    )
    assert response.status_code == 200
    # Response is SSE, so we read text
    text = response.text
    assert "We offer service1 and service2" in text

    # ---------------------------------------------------------
    # 5. Chat API - Place Order
    # ---------------------------------------------------------
    async def mock_order_stream(*args, **kwargs):
        yield {"order_approval": {"messages": [AIMessage(content="Order confirmed successfully!")]}}

    mock_astream.side_effect = mock_order_stream
    
    response = client.post(
        f"/keystone/business/{business_id}/chat/{chat_id}",
        json={"message": "I want to order service1 for John"}
    )
    assert response.status_code == 200
    text = response.text
    assert "Order confirmed successfully!" in text

    # ---------------------------------------------------------
    # 6. Get Chat API (History)
    # ---------------------------------------------------------
    mock_state = MagicMock()
    mock_state.values = {
        "messages": [
            HumanMessage(content="What services do you offer?"),
            AIMessage(content="We offer service1 and service2"),
            HumanMessage(content="I want to order service1 for John"),
            AIMessage(content="Order confirmed successfully!")
        ]
    }
    
    async def get_state_mock(*args, **kwargs):
        return mock_state
        
    mock_aget_state.side_effect = get_state_mock

    response = client.get(f"/keystone/business/{business_id}/chat/{chat_id}")
    assert response.status_code == 200
    history = response.json()["messages"]
    assert len(history) == 4
    assert history[0]["content"] == "What services do you offer?"
    assert history[1]["content"] == "We offer service1 and service2"
    assert history[2]["content"] == "I want to order service1 for John"
    assert history[3]["content"] == "Order confirmed successfully!"
