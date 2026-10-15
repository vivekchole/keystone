import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.business import Business
from app.core.database import SessionLocal, get_db


@pytest.fixture(autouse=True)
def clean_db_overrides():
    saved = app.dependency_overrides.get(get_db)
    app.dependency_overrides.pop(get_db, None)
    yield
    if saved:
        app.dependency_overrides[get_db] = saved


@pytest.fixture
def client():
    return TestClient(app)


def test_ui_endpoints(client):
    # Test GET / returns index.html
    res = client.get("/")
    assert res.status_code == 200
    assert "Keystone AI Chatbot & Business Management Portal" in res.text

    # Test GET /ui returns index.html
    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "Keystone AI Chatbot & Business Management Portal" in res_ui.text


def test_business_list_and_update(client):
    db = SessionLocal()
    try:
        biz = db.query(Business).filter(Business.id == "test-biz-1").first()
        if not biz:
            biz = Business(
                id="test-biz-1",
                name="Test Pizzeria",
                services=["Pizza", "Coke"],
                timings="9am - 9pm",
                description="Delicious artisanal pizzas.",
            )
            db.add(biz)
            db.commit()
    finally:
        db.close()

    # 1. Test GET /keystone/business/
    res = client.get("/keystone/business/")
    assert res.status_code == 200
    businesses = res.json()
    assert isinstance(businesses, list)
    assert any(b["id"] == "test-biz-1" for b in businesses)

    # 2. Test PUT /keystone/business/{business_id}
    update_data = {
        "name": "Updated Test Pizzeria",
        "services": ["Pizza", "Pasta", "Gelato"],
        "timings": "10am - 11pm",
        "description": "Authentic Italian wood-fired ovens.",
    }
    put_res = client.put("/keystone/business/test-biz-1", json=update_data)
    assert put_res.status_code == 200
    updated = put_res.json()
    assert updated["name"] == "Updated Test Pizzeria"
    assert "Gelato" in updated["services"]


def test_admin_overview_api(client):
    res = client.get("/keystone/admin/overview")
    assert res.status_code == 200
    data = res.json()
    assert "stats" in data
    assert "businesses" in data
    assert "orders" in data
    assert "customers" in data
    assert "total_businesses" in data["stats"]
    assert "total_orders" in data["stats"]
    assert "total_customers" in data["stats"]
