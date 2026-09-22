import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

class ProductVisionDetector:
    """
    Klasa odpowiedzialna za analizę obrazów produktów przy użyciu 
    splotowej sieci neuronowej (CNN) w oparciu o architekturę ResNet-18.
    """
    def __init__(self):
        # 1. Wybór urządzenia obliczeniowego (GPU CUDA jeśli dostępne, w przeciwnym razie CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisionDetector] Inicjalizacja modelu na urządzeniu: {self.device}")

        # 2. Załadowanie gotowych, wytrenowanych wag (Transfer Learning na zbiorze ImageNet)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()  # Przełączenie sieci w tryb predykcji/ewaluacji (wyłączenie Dropout / BatchNorm)

        # 3. Przygotowanie pipeline'u transformacji obrazu (zgodnie z wymaganiami sieci ResNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Skalowanie obrazu do wymiarów 224x224 px
            transforms.ToTensor(),          # Konwersja z obrazu PIL/numpy do Tenzora PyTorch
            transforms.Normalize(           # Normalizacja wartości pikseli (standaryzacja ImageNet)
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Meta-etykiety klas zbioru ImageNet
        self.categories = self.weights.meta["categories"]

    def classify_image(self, image_path: str) -> dict:
        """
        Pobiera ścieżkę do pliku graficznego, przekształca go, przechodzi przez sieć neuronową 
        i zwraca najbardziej prawdopodobne kategorie wraz z poziomem pewności.
        """
        if not os.path.exists(image_path):
            # Tryb zastępczy gdy plik fizycznie nie istnieje w katalogu raw
            return {
                "status": "mock_result",
                "message": f"Plik {image_path} nie istnieje. Zwrócono wynik symulowany.",
                "top_prediction": "electronic_equipment",
                "confidence": 0.92
            }

        try:
            # 1. Wczytanie obrazu i konwersja do przestrzeni RGB
            image = Image.open(image_path).convert('RGB')
            
            # 2. Zastosowanie transformacji i dodanie wymiaru batcha (shape: [1, 3, 224, 224])
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)

            # 3. Przeprowadzenie predykcji bez obliczania gradientów (oszczędność pamięci RAM/VRAM)
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
    # Test działania na przykładowym pliku
    sample_img_path = "data/raw/images/img_0001.jpg"
    result = detector.classify_image(sample_img_path)
    print("\n--- Wynik Analizy Obrazu (PyTorch Computer Vision) ---")
    print(result)