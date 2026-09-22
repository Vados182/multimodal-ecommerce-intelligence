import os

# Ograniczenie narzutu pamięciowego PyTorcha na CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

class ProductVisionDetector:
    """
    Klasa odpowiedzialna za analizę obrazów produktów przy użyciu 
    splotowej sieci neuronowej (CNN) w oparciu o architekturę ResNet-18.
    Zoptymalizowana pod kątem niskiego zużycia pamięci RAM (np. 512MB limit na Renderze).
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.weights = None
        self.categories = None

        # Przygotowanie pipeline'u transformacji obrazu
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        """Leniwe ładowanie modelu ResNet-18 do pamięci RAM przy pierwszym użyciu."""
        if self.model is None:
            print(f"[VisionDetector] Inicjalizacja modelu na urządzeniu: {self.device}")
            with torch.no_grad():
                self.weights = ResNet18_Weights.DEFAULT
                self.model = resnet18(weights=self.weights)
                self.model.to(self.device)
                self.model.eval()
                self.categories = self.weights.meta["categories"]

    def classify_image(self, image_path: str) -> dict:
        """
        Pobiera ścieżkę do pliku graficznego, przekształca go, przechodzi przez sieć neuronową 
        i zwraca najbardziej prawdopodobne kategorie wraz z poziomem pewności.
        """
        if not os.path.exists(image_path):
            return {
                "status": "mock_result",
                "message": f"Plik {image_path} nie istnieje. Zwrócono wynik symulowany.",
                "top_prediction": "electronic_equipment",
                "confidence": 0.92
            }

        try:
            # Ładowanie modelu do pamięci dopiero przy pierwszym zapytaniu
            self._load_model()

            # 1. Wczytanie obrazu i konwersja do przestrzeni RGB
            image = Image.open(image_path).convert('RGB')
            
            # 2. Zastosowanie transformacji i dodanie wymiaru batcha
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)

            # 3. Przeprowadzenie predykcji bez obliczania gradientów
            with torch.no_grad():
                outputs = self.model(tensor_image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # 4. Pobranie najwyższego wyniku (Top-1 prediction)
            top_prob, top_cat_id = torch.topk(probabilities, 1)
            predicted_label = self.categories[top_cat_id[0].item()]
            confidence = float(top_prob[0].item())

            return {
                "status": "success",
                "top_prediction": predicted_label,
                "confidence": round(confidence, 4)
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    detector = ProductVisionDetector()
    sample_img_path = "data/raw/images/img_0001.jpg"
    result = detector.classify_image(sample_img_path)
    print("\n--- Wynik Analizy Obrazu (PyTorch Computer Vision) ---")
    print(result)