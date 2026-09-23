import os

# Ograniczenie narzutu pamięciowego PyTorcha na CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

class ProductVisionDetector:
    """
    Ultra-lekka klasa do analizy obrazów oparta o MobileNetV3-Small.
    Idealna do środowisk z ograniczeniem < 512MB RAM.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.weights = None
        self.categories = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        """Leniwe ładowanie ultra-lekkiego modelu."""
        if self.model is None:
            print(f"[VisionDetector] Inicjalizacja MobileNetV3 na: {self.device}")
            with torch.no_grad():
                self.weights = MobileNet_V3_Small_Weights.DEFAULT
                self.model = mobilenet_v3_small(weights=self.weights)
                self.model.to(self.device)
                self.model.eval()
                self.categories = self.weights.meta["categories"]

    def classify_image(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {
                "status": "mock_result",
                "message": f"Plik {image_path} nie istnieje.",
                "top_prediction": "electronic_equipment",
                "confidence": 0.92
            }

        try:
            self._load_model()

            image = Image.open(image_path).convert('RGB')
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

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