# 1. Obraz bazowy z instalacją Pythona 3.11
FROM python:3.11-slim

# 2. Zmienne środowiskowe zapobiegające tworzeniu plików .pyc i buforowaniu logów
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Instalacja pakietów systemowych OS (apt-get):
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Ustawienie katalogu roboczego
WORKDIR /app

# 5. Kopiowanie i instalacja zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Pre-download wag PyTorch na etapie budowania kontenera (przed COPY . .)
RUN python -c "import torchvision.models as models; models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)"

# 7. Kopiowanie reszty kodu aplikacji (zmienia się najczęściej, więc jest na końcu)
COPY . .

# 8. Port informacyjny dla Dockera
EXPOSE $PORT

# 9. Uruchomienie serwera Uvicorn z dynamicznym portem z usługi Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]