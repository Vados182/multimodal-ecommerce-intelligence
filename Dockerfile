# 1. Obraz bazowy: Wybieramy smukłą wersję Pythona 3.11 na Linuksie (Debian).
# Słowo "slim" oznacza zminimalizowany obraz, co przyspiesza budowanie w chmurze Render.
FROM python:3.11-slim

# 2. Zmienne środowiskowe dla Pythona:
# PYTHONUNBUFFERED=1 - wymusza natychmiastowe wypisywanie logów w terminalu (bez buforowania).
# PYTHONDONTWRITEBYTECODE=1 - zapobiega tworzeniu zbędnych plików .pyc w kontenerze.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Instalacja pakietów systemowych OS (apt-get):
# Biblioteki C/C++ oraz graficzne (libgl1, libglib) są wymagane przez PyTorch i OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Ustalamy katalog roboczy wewnątrz kontenera, do którego będą trafiać pliki projektu.
WORKDIR /app

# 5. Kopiujemy najpierw sam plik requirements.txt i instalujemy zależności.
# Dzięki temu Docker wykorzysta swój cache i nie będzie ponownie pobierał paczek przy zmianie samych skryptów .py.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Kopiujemy cały pozostały kod źródłowy z Twojego komputera do folderu /app w kontenerze.
COPY . .

# 7. Deklaracja portu informacyjna dla Dockera (Render i tak dynamicznie przydziela swój port przez zmienną $PORT).
EXPOSE $PORT

# 8. Polecenie startowe (CMD):
# Uruchamia serwer Uvicorn, pobierając port przydzielony przez Render (zmienna ${PORT}).
# Jeśli zmienna nie istnieje (np. lokalnie), użyje domyślnego portu 10000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]