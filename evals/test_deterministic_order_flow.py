import os
import uuid
import json
import pytest
import httpx
import dotenv
from sqlalchemy import text
from app.core.database import SessionLocal
from app.models.order import Order

dotenv.load_dotenv()

BASE_URL = "http://127.0.0.1:8000"


def parse_sse_events(response_text: str):
    """Parse raw SSE stream into a list of parsed JSON event payloads."""
    events = []
    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            raw_data = line[len("data:"):].strip()
            if raw_data == "[DONE]":
                continue
            try:
                events.append(json.loads(raw_data))
            except Exception:
                events.append({"raw": raw_data})
    return events


@pytest.fixture
def api_client():
    """Client for testing against the running FastAPI application."""
    try:
        # Check if local server is listening
        res = httpx.get(f"{BASE_URL}/", timeout=2.0)
        if res.status_code == 200:
            with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
                yield client
                return
    except Exception:
        pass

    # Fallback to in-process ASGI client if standalone server isn't running
    from app.main import app
    with httpx.Client(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        yield client


@pytest.mark.eval
def test_order_flow_deterministic_lifecycle(api_client):
    """
    Deterministic End-to-End Evaluation of the Order Flow:
    1. Business Creation in PostgreSQL
    2. Auth Gatekeeper: Phone Challenge -> OTP Challenge -> Verification
    3. Order Intent & Customer Profile Lookup (Name, Address)
    4. Human-in-the-Loop Interrupt Gate
    5. Order Approval & Database Commitment with Idempotency Key
    6. Database Assertions (Order row, columns, JSON items)
    7. Duplicate Prevention / Idempotency Re-check
    """
    # -------------------------------------------------------------
    # Step 1: Create Business
    # -------------------------------------------------------------
    biz_name = f"Deterministic Pizza {uuid.uuid4().hex[:6]}"
    biz_payload = {
        "name": biz_name,
        "services": ["Wood-fired Pizza", "Garlic Bread", "Delivery"],
        "timings": "11:00 AM to 11:00 PM Daily",
        "description": "Authentic stone-oven Italian pizzeria."
    }
    res = api_client.post("/keystone/business/", json=biz_payload)
    assert res.status_code == 200, f"Business creation failed: {res.text}"
    biz_id = res.json()["id"]

    chat_id = f"eval_det_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------
    # Step 2: Auth Gatekeeper - Unauthenticated user must get phone prompt
    # -------------------------------------------------------------
    res1 = api_client.post(
        f"/keystone/business/{biz_id}/chat/{chat_id}",
        json={"message": "Hello, I want to order 2 pizzas."}
    )
    assert res1.status_code == 200
    events1 = parse_sse_events(res1.text)
    combined_1 = " ".join([e.get("content", "") for e in events1]).lower()
    assert any(term in combined_1 for term in ["phone", "mobile", "number"]), (
        f"Step 2 failed: Expected mobile phone prompt, got: {combined_1}"
    )

    # -------------------------------------------------------------
    # Step 3: Auth Gatekeeper - Provide phone number (registered demo profile)
    # -------------------------------------------------------------
    res2 = api_client.post(
        f"/keystone/business/{biz_id}/chat/{chat_id}",
        json={"message": "9876543210"}
    )
    assert res2.status_code == 200
    events2 = parse_sse_events(res2.text)
    combined_2 = " ".join([e.get("content", "") for e in events2]).lower()
    assert any(term in combined_2 for term in ["otp", "code", "1234"]), (
        f"Step 3 failed: Expected OTP prompt, got: {combined_2}"
    )

    # -------------------------------------------------------------
    # Step 4: Auth Gatekeeper - Submit OTP and confirm authentication
    # -------------------------------------------------------------
    res3 = api_client.post(
        f"/keystone/business/{biz_id}/chat/{chat_id}",
        json={"message": "1234"}
    )
    assert res3.status_code == 200
    events3 = parse_sse_events(res3.text)
    combined_3 = " ".join([e.get("content", "") for e in events3]).lower()
    assert any(term in combined_3 for term in ["otp verified", "welcome", "successful"]), (
        f"Step 4 failed: Expected successful authentication, got: {combined_3}"
    )

    # -------------------------------------------------------------
    # Step 5: Place Order - Profile lookup & Interrupt Trigger
    # -------------------------------------------------------------
    res4 = api_client.post(
        f"/keystone/business/{biz_id}/chat/{chat_id}",
        json={"message": "I want to place an order for 2 Margherita Pizzas and 1 Garlic Bread to my registered address please."}
    )
    assert res4.status_code == 200
    events4 = parse_sse_events(res4.text)

    # Check for either immediate interrupt or summary asking for confirmation
    has_interrupt = any(e.get("node") == "__interrupt__" or e.get("approval_required") for e in events4)
    if not has_interrupt:
        # Agent asked for confirmation, respond with "yes"
        res4_confirm = api_client.post(
            f"/keystone/business/{biz_id}/chat/{chat_id}",
            json={"message": "Yes, details are correct. Proceed with the order."}
        )
        assert res4_confirm.status_code == 200
        events4 = parse_sse_events(res4_confirm.text)
        has_interrupt = any(e.get("node") == "__interrupt__" or e.get("approval_required") for e in events4)

    assert has_interrupt, "Step 5 failed: Order flow must trigger a human approval interrupt gate!"

    # -------------------------------------------------------------
    # Step 6: Approve Order & Commit to Database
    # -------------------------------------------------------------
    idempotency_key = f"eval_idemp_{chat_id}"
    res_approve = api_client.post(
        f"/keystone/business/{biz_id}/chat/{chat_id}/approve",
        json={"approved": True, "idempotency_key": idempotency_key}
    )
    assert res_approve.status_code == 200
    approve_events = parse_sse_events(res_approve.text)
    order_confirmation = " ".join([e.get("content", "") for e in approve_events])
    assert "order placed with id" in order_confirmation.lower() or "success" in order_confirmation.lower(), (
        f"Step 6 failed: Expected order confirmation event, got: {order_confirmation}"
    )

    # -------------------------------------------------------------
    # Step 7: Deterministic PostgreSQL Verification
    # -------------------------------------------------------------
    db = SessionLocal()
    try:
        db_order = (
            db.query(Order)
            .filter(Order.idempotency_key == idempotency_key)
            .first()
        )
        assert db_order is not None, f"Order with idempotency_key '{idempotency_key}' was not found in DB!"
        assert str(db_order.business_id) == biz_id
        assert db_order.customer_name == "Vivek"
        assert len(db_order.items) >= 2
        print(f"\n[DB ASSERTION PASSED]: Order {db_order.id} verified in PostgreSQL.")
    finally:
        db.close()

    # -------------------------------------------------------------
    # Step 8: Verify Idempotency Duplicate Replay
    # -------------------------------------------------------------
    res_dup = api_client.post(
        f"/keystone/business/{biz_id}/chat/{chat_id}/approve",
        json={"approved": True, "idempotency_key": idempotency_key}
    )
    assert res_dup.status_code == 200
    dup_data = res_dup.json()
    assert "already placed previously" in dup_data.get("message", "").lower(), (
        f"Expected idempotent deduplication response, got: {dup_data}"
    )

    # Verify database still has exactly 1 order with this idempotency key
    db = SessionLocal()
    try:
        count = db.query(Order).filter(Order.idempotency_key == idempotency_key).count()
        assert count == 1, f"Expected exactly 1 order for key '{idempotency_key}', found {count}!"
        print(f"[IDEMPOTENCY ASSERTION PASSED]: Deduplicated safely. Exactly {count} order in DB.")
    finally:
        db.close()
