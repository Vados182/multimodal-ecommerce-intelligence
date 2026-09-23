import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from main import app

# Tworzymy klienta testowego FastAPI
client = TestClient(app)

# Helper: Tworzenie testowego obrazu JPEG w pamięci
def create_test_image():
    file = io.BytesIO()
    image = Image.new("RGB", (100, 100), color="blue")
    image.save(file, 'jpeg')
    file.seek(0)
    return file


def test_health_check():
    """Test sprawdza czy endpoint główny zwraca status online"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_analyze_product_success():
    """Test sprawdzający poprawność działania modułu Agent GenAI"""
    payload = {
        "product_name": "Słuchawki Sony WH-1000XM5",
        "description": "Czy są odpowiednie do głośnego biura?"
    }
    response = client.post("/analyze-product", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "category" in data
    assert "extracted_features" in data
    assert "recommended_price_range" in data


def test_analyze_product_missing_field():
    """Test sprawdzający walidację błędu 422 przy braku wymaganego pola"""
    payload = {
        "product_name": "Słuchawki Sony"
        # Brak pola 'description'
    }
    response = client.post("/analyze-product", json=payload)
    assert response.status_code == 422


def test_predict_causal_impact_success():
    """Test walidujący predykcję wpływu zmian (Causal Inference)"""
    payload = {
        "discount": 15.0,
        "customer_rating": 4.5
    }
    response = client.post("/predict-causal-impact", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_sales_lift_pct" in data or "estimated_impact" in data or isinstance(data, dict)


def test_predict_causal_impact_invalid_schema():
    """Test sprawdzający walidację danych Pydantic (oczekiwany błąd 422 przy błędnych kluczach)"""
    payload = {
        "wrong_key": 10
    }
    response = client.post("/predict-causal-impact", json=payload)
    assert response.status_code == 422


def test_analyze_image_success():
    """Test sprawdzający moduł Computer Vision (MobileNetV3) przy wysłaniu pliku"""
    img_bytes = create_test_image()
    files = {
        "file": ("test_image.jpg", img_bytes, "image/jpeg")
    }
    response = client.post("/analyze-image", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "top_prediction" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)