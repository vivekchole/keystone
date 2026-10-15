import os
import sys
from unittest.mock import MagicMock, patch

os.environ["LLM_MODEL_NAME"] = "gpt-4o-mini"
os.environ["LLM_MODEL_KEY"] = "mock-key"

mock_saver = MagicMock()
sys.modules["langgraph.checkpoint.postgres.aio"] = MagicMock()
sys.modules["langgraph.checkpoint.postgres.aio"].AsyncPostgresSaver = MagicMock(
    return_value=mock_saver
)

import pytest
from app.langgraph.nodes import get_customer_profile, get_order, take_order
from app.core.security import sanitize_error_message


def test_get_customer_profile_invalid_input():
    """Verify input validation on get_customer_profile."""
    res1 = get_customer_profile.invoke({"phone_number": ""})
    assert "Please provide a valid phone number" in res1

    res2 = get_customer_profile.invoke({"phone_number": "   "})
    assert "Please provide a valid phone number" in res2

    res3 = get_customer_profile.invoke({"phone_number": "abc-def"})
    assert "No customer profile found for phone number: abc-def" in res3


def test_get_customer_profile_db_error_isolation():
    """Verify that database errors during profile lookup return a safe message without leaking internal details."""
    mock_db = MagicMock()
    mock_db.query.side_effect = Exception("FATAL: password authentication failed for user 'postgres'")

    with patch("app.langgraph.nodes.SessionLocal", return_value=mock_db):
        res = get_customer_profile.invoke({"phone_number": "9876543210"})
        assert "Unable to retrieve customer profile due to a temporary service error" in res
        assert "password" not in res
        assert "postgres" not in res


def test_get_order_invalid_input():
    """Verify input validation on get_order."""
    res1 = get_order.invoke({"order_id": ""})
    assert "Please provide a valid Order ID" in res1

    res2 = get_order.invoke({"order_id": "   "})
    assert "Please provide a valid Order ID" in res2


def test_get_order_db_error_isolation():
    """Verify that database errors during order lookup return a safe message without leaking internal details."""
    mock_db = MagicMock()
    mock_db.query.side_effect = Exception("OperationalError: connection to server at 'db.internal.cloud' failed")

    with patch("app.langgraph.nodes.SessionLocal", return_value=mock_db):
        res = get_order.invoke({"order_id": "order-123"})
        assert "Unable to retrieve order details due to a temporary service error" in res
        assert "db.internal.cloud" not in res
        assert "OperationalError" not in res


def test_take_order_invalid_input():
    """Verify validation on customer_name and items in take_order."""
    # Empty customer name
    res1 = take_order.invoke({"customer_name": "", "items": ["Pizza"]})
    assert "Customer name is required" in res1

    # Whitespace customer name
    res2 = take_order.invoke({"customer_name": "   ", "items": ["Pizza"]})
    assert "Customer name is required" in res2

    # Empty items
    res3 = take_order.invoke({"customer_name": "Alice", "items": []})
    assert "At least one item is required" in res3

    # Items containing only whitespace
    res4 = take_order.invoke({"customer_name": "Alice", "items": ["  ", ""]})
    assert "At least one valid item is required" in res4


def test_take_order_db_error_isolation():
    """Verify that database errors during take_order trigger rollback and return a safe message."""
    mock_db = MagicMock()
    # Pre-check returns None
    filter_mock = MagicMock()
    filter_mock.first.return_value = None
    mock_db.query.return_value.filter.return_value = filter_mock

    # Commit raises DB exception containing sensitive credentials
    mock_db.commit.side_effect = Exception("postgresql://admin:secret123@localhost/keystone: connection lost")

    with patch("app.langgraph.nodes.SessionLocal", return_value=mock_db):
        res = take_order.invoke({
            "customer_name": "Bob",
            "items": ["Burger"],
            "business_id": "biz-1",
            "idempotency_key": "key-err-test",
        })
        mock_db.rollback.assert_called_once()
        assert "Error placing order: A database error occurred" in res
        assert "secret123" not in res
        assert "postgresql://" not in res


def test_sanitize_error_message():
    """Verify that sanitize_error_message redacts connection strings, secrets, and system paths."""
    sensitive_err = Exception("Failed connecting to postgresql://user:my_secret_pass@127.0.0.1:5432/db")
    sanitized = sanitize_error_message(sensitive_err)
    assert sanitized == "An internal service error occurred. Please try again later."
    assert "my_secret_pass" not in sanitized

    path_err = Exception("FileNotFoundError: [Errno 2] No such file: '/Users/admin/keys/secret.json'")
    sanitized_path = sanitize_error_message(path_err)
    assert sanitized_path == "An internal service error occurred. Please try again later."

    normal_err = Exception("Invalid input format")
    assert sanitize_error_message(normal_err) == "Invalid input format"
