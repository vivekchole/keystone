import asyncio
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
from app.main import app
from app.core.database import get_db
from app.core.lock_manager import ThreadLockManager, thread_lock_manager
from app.models.business import Business

client = TestClient(app)

mock_business = Business(
    id="biz-lock-test",
    name="Lock Test Business",
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


@pytest.mark.asyncio
async def test_thread_lock_manager_acquire_and_release():
    mgr = ThreadLockManager()
    key = "session_1"

    # Acquire succeeds
    assert await mgr.acquire(key, timeout=0.5) is True
    assert mgr.is_locked(key) is True
    assert mgr._ref_counts[key] == 1

    # Second acquire times out
    assert await mgr.acquire(key, timeout=0.1) is False
    # Ref count remains 1 after timeout cleanup
    assert mgr._ref_counts[key] == 1

    # Release cleans up dictionary
    await mgr.release(key)
    assert mgr.is_locked(key) is False
    assert key not in mgr._locks
    assert key not in mgr._ref_counts


@pytest.mark.asyncio
async def test_thread_lock_manager_queuing():
    mgr = ThreadLockManager()
    key = "session_queue"
    order = []

    async def worker(worker_id: int, hold_time: float):
        acquired = await mgr.acquire(key, timeout=2.0)
        assert acquired is True
        order.append(f"start_{worker_id}")
        await asyncio.sleep(hold_time)
        order.append(f"end_{worker_id}")
        await mgr.release(key)

    t1 = asyncio.create_task(worker(1, 0.15))
    await asyncio.sleep(0.02)
    t2 = asyncio.create_task(worker(2, 0.05))

    await asyncio.gather(t1, t2)
    assert order == ["start_1", "end_1", "start_2", "end_2"]
    assert key not in mgr._locks


def test_concurrent_chat_conflict_409():
    """
    Test that when an active stream holds the lock for a chat_id,
    a concurrent request for the same chat_id receives HTTP 409 Conflict.
    """
    chat_id = "lock-conflict-thread"
    orig_timeout = thread_lock_manager._default_timeout
    thread_lock_manager._default_timeout = 0.1

    # Acquire lock beforehand to simulate an in-flight request
    asyncio.run(thread_lock_manager.acquire(chat_id))
    try:
        response = client.post(
            f"/keystone/business/{mock_business.id}/chat/{chat_id}",
            json={"message": "Concurrent message while busy"},
        )
        assert response.status_code == 409
        detail = response.json().get("detail", "")
        assert "already being processed" in detail
    finally:
        asyncio.run(thread_lock_manager.release(chat_id))
        thread_lock_manager._default_timeout = orig_timeout

    # Once released, subsequent request should proceed
    async def mock_stream(*args, **kwargs):
        yield {"node": "test", "messages": []}

    with patch("app.api.chat.graph.astream", side_effect=mock_stream):
        response_after = client.post(
            f"/keystone/business/{mock_business.id}/chat/{chat_id}",
            json={"message": "Message after lock released"},
        )
        assert response_after.status_code == 200
        assert "data: [DONE]" in response_after.text
        # Verify lock was cleanly released
        assert thread_lock_manager.is_locked(chat_id) is False


def test_concurrent_approve_conflict_409():
    """
    Test that when a thread lock is held, approval endpoint returns 409 Conflict.
    """
    chat_id = "approve-lock-conflict-thread"
    orig_timeout = thread_lock_manager._default_timeout
    thread_lock_manager._default_timeout = 0.1

    asyncio.run(thread_lock_manager.acquire(chat_id))
    try:
        response = client.post(
            f"/keystone/business/{mock_business.id}/chat/{chat_id}/approve",
            json={"approved": True},
        )
        assert response.status_code == 409
        assert "already being processed" in response.json().get("detail", "")
    finally:
        asyncio.run(thread_lock_manager.release(chat_id))
        thread_lock_manager._default_timeout = orig_timeout
        assert thread_lock_manager.is_locked(chat_id) is False

