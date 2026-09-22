from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil

# Importy z naszych modułów
from src.agents.product_agent import ProductAgent
from src.models.vision_detector import ProductVisionDetector

# 1. Inicjalizacja głównej aplikacji FastAPI z metadanymi
app = FastAPI(
    title="Multimodal E-Commerce Intelligence API",
    description="Multimodalne API łączące analizę tekstu (LLM), obrazu (PyTorch ResNet) oraz wnioskowanie przyczynowe.",
    version="1.0.0"
)

# 2. Inicjalizacja modeli w pamięci podręcznej (Singleton)
agent = ProductAgent()
vision_detector = ProductVisionDetector()


# 3. Schematy Pydantic do walidacji danych wejściowych
class ProductRequest(BaseModel):
    description: str


class CausalRequest(BaseModel):
    discount: float
    customer_rating: float


# --- ENDPOINTY API ---

@app.get("/")
def read_root():
    """
    Endpoint główny - zwraca status serwisu oraz spis dostępnych ścieżek.
    """
    return {
        "status": "online",
        "service": "Multimodal E-Commerce Intelligence API",
        "endpoints": [
            "/health",
            "/analyze-product",
            "/predict-causal-impact",
            "/analyze-image"
        ]
    }


@app.get("/health")
def health_check():
    """
    Health check używany m.in. przez Render / Kubernetes do weryfikacji żywotności aplikacji.
    """
    return {"status": "ok", "message": "Service is running on Render"}


@app.post("/analyze-product")
def analyze_product_endpoint(request: ProductRequest):
    """
    Analiza opisu tekstu oferty przy użyciu Agenta GenAI (LangChain + OpenAI / Mock).
    """
    result = agent.analyze_product(request.description)
    return {
        "category": result.category,
        "extracted_features": result.extracted_features,
        "recommended_price_range": result.recommended_price_range,
        "confidence_score": result.confidence_score
    }


@app.post("/predict-causal-impact")
def predict_causal_impact(request: CausalRequest):
    """
    Wnioskowanie przyczynowe - szacuje wpływ wysokości rabatu na wolumen sprzedaży.
    """
    estimated_impact = (request.discount * 0.0350) + (request.customer_rating * -0.0258)
    return {
        "applied_discount": request.discount,
        "customer_rating": request.customer_rating,
        "estimated_quantity_lift": round(estimated_impact, 4),
        "recommendation": "Rabat przyniesie dodatni skutek wolumenowy" if estimated_impact > 0 else "Niska efektywność rabatu"
    }


@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint do wgrywania plików graficznych i klasyfikacji ich za pomocą PyTorch ResNet-18.
    """
    # Tymczasowy zapis wgranego pliku na dysku
    temp_dir = "data/raw/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Klasyfikacja za pomocą naszego modelu PyTorch
    prediction = vision_detector.classify_image(temp_file_path)

    # Usunięcie pliku tymczasowego
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return prediction