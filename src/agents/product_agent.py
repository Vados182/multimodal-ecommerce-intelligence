import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Struktura danych wyjściowych z agenta (Pydantic Schema)
class ProductAnalysisResult(BaseModel):
    category: str = Field(description="Zarekomendowana kategoria produktu (np. Electronics, Automotive itp.)")
    extracted_features: List[str] = Field(description="Lista kluczowych cech wyciągniętych z opisu")
    recommended_price_range: str = Field(description="Sugerowany widełek cenowy, np. '100-150 PLN'")
    confidence_score: float = Field(description="Pewność kategoryzacji w skali od 0.0 do 1.0")

class ProductAgent:
    def __init__(self, api_key: Optional[str] = None):
        # Pobranie klucza ze zmiennych środowiskowych lub domyślnego parametru
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=self.api_key)
            self.structured_llm = self.llm.with_structured_output(ProductAnalysisResult)
        else:
            self.llm = None
            print("Ostrzeżenie: Brak klucza OPENAI_API_KEY. Agent będzie działał w trybie Symulacji (Mock).")

    def analyze_product(self, raw_description: str) -> ProductAnalysisResult:
        if not self.llm:
            # Tryb awaryjny / symulacji gdy nie ma podanego klucza API
            return ProductAnalysisResult(
                category="Electronics",
                extracted_features=["Stan idealny", "Szybka wysyłka", "Zastosowanie domowe"],
                recommended_price_range="100-250 PLN",
                confidence_score=0.95
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Jesteś ekspertem e-commerce ds. automatycznej kategoryzacji i analizy parametrów ofert."),
            ("human", "Przeanalizuj poniższy opis oferty i wyciągnij z niego strukturyzowane dane:\n\n{description}")
        ])

        chain = prompt | self.structured_llm
        return chain.invoke({"description": raw_description})

if __name__ == "__main__":
    agent = ProductAgent()
    sample_text = "Super okazja! Komputer stacjonarny PRD-0001. Stan idealny, szybka wysyłka. Idealne do biura i gier."
    result = agent.analyze_product(sample_text)
    
    print("\n--- Wynik analizy Agenta AI ---")
    print(f"Kategoria: {result.category}")
    print(f"Cechy: {result.extracted_features}")
    print(f"Sugerowana cena: {result.recommended_price_range}")
    print(f"Pewność: {result.confidence_score}")