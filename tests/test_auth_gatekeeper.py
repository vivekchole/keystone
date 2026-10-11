import os
import sys
from unittest.mock import MagicMock
from langgraph.checkpoint.memory import MemorySaver

os.environ["LLM_MODEL_NAME"] = "gpt-4o-mini"
os.environ["LLM_MODEL_KEY"] = "mock-key"

mock_saver = MemorySaver()
sys.modules['langgraph.checkpoint.postgres.aio'] = MagicMock()
sys.modules['langgraph.checkpoint.postgres.aio'].AsyncPostgresSaver = MagicMock(return_value=mock_saver)

import pytest
from langchain_core.messages import HumanMessage
from app.langgraph.nodes import auth_gatekeeper_node


def test_auth_gatekeeper_flow():
    # 1. Unauthenticated user with no phone provided
    state_turn1 = {
        "messages": [HumanMessage(content="Hello, I want to chat")],
        "is_authenticated": False,
        "phone_number": None,
        "otp": None,
        "otp_attempts": 0,
    }
    res1 = auth_gatekeeper_node(state_turn1)
    assert "messages" in res1
    assert "please provide your 10-digit mobile number" in res1["messages"][0].content

    # 2. User provides 10-digit mobile number -> Send OTP (1234)
    state_turn2 = {
        "messages": [HumanMessage(content="9876543210")],
        "is_authenticated": False,
        "phone_number": None,
        "otp": None,
        "otp_attempts": 0,
    }
    res2 = auth_gatekeeper_node(state_turn2)
    assert res2["phone_number"] == "9876543210"
    assert res2["otp"] == "1234"
    assert "Demo OTP: 1234" in res2["messages"][0].content

    # 3. User enters wrong OTP (Attempt 1 failed)
    state_turn3 = {
        "messages": [HumanMessage(content="9999")],
        "is_authenticated": False,
        "phone_number": "9876543210",
        "otp": "1234",
        "otp_attempts": 0,
    }
    res3 = auth_gatekeeper_node(state_turn3)
    assert res3["otp_attempts"] == 1
    assert "2 attempt(s) remaining" in res3["messages"][0].content

    # 4. User enters wrong OTP (Attempt 2 failed)
    state_turn4 = {
        "messages": [HumanMessage(content="8888")],
        "is_authenticated": False,
        "phone_number": "9876543210",
        "otp": "1234",
        "otp_attempts": 1,
    }
    res4 = auth_gatekeeper_node(state_turn4)
    assert res4["otp_attempts"] == 2
    assert "1 attempt(s) remaining" in res4["messages"][0].content

    # 5. User enters wrong OTP 3rd time -> reset
    state_turn5 = {
        "messages": [HumanMessage(content="7777")],
        "is_authenticated": False,
        "phone_number": "9876543210",
        "otp": "1234",
        "otp_attempts": 2,
    }
    res5 = auth_gatekeeper_node(state_turn5)
    assert res5["phone_number"] is None
    assert "exceeded the maximum of 3 OTP attempts" in res5["messages"][0].content

    # 6. User enters correct OTP -> Verified!
    state_turn6 = {
        "messages": [HumanMessage(content="1234")],
        "is_authenticated": False,
        "phone_number": "9876543210",
        "otp": "1234",
        "otp_attempts": 0,
    }
    res6 = auth_gatekeeper_node(state_turn6)
    assert res6["is_authenticated"] is True
    assert "OTP verified successfully" in res6["messages"][0].content
