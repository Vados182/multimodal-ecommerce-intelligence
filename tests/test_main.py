import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test sprawdzający status serwera."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

def test_analyze_product_success():
    """Test sprawdzający poprawność zapytania do Agenta."""
    payload = {
        "product_name": "Słuchawki Sony WH-1000XM5",
        "description": "Czy nadają się do głośnego biura?"
    }
    response = client.post("/analyze-product", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence_score" in data

def test_analyze_product_missing_field():
    """Test sprawdzający walidację błędu 422 przy braku wymaganego pola."""
    payload = {
        "product_name": "Słuchawki Sony"
        # Brak pola 'description'
    }
    response = client.post("/analyze-product", json=payload)
    assert response.status_code == 422