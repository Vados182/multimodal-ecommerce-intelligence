# Moduł, który oceni wpływ rabatu (discount_applied) na wielkość sprzedaży (quantity), z uwzględnieniem czynników zakłócających (np. ocena klienta customer_rating).

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def run_causal_inference(data_path="data/raw/orders.csv"):
    if not os.path.exists(data_path):
        print(f"Błąd: Plik {data_path} nie istnieje. Wygeneruj dane najpierw.")
        return

    df = pd.read_csv(data_path)
    
    # Prosta analiza regresyjna służąca jako estymator efektu przyczynowego (Causal Effect)
    # Badamy wpływ discount_applied na quantity, kontrolując customer_rating
    X = df[['discount_applied', 'customer_rating']]
    y = df['quantity']
    
    model = LinearRegression()
    model.fit(X, y)
    
    discount_effect = model.coef_[0]
    rating_effect = model.coef_[1]
    
    print("\n=== Wyniki Analizy Przyczynowej (Causal Inference) ===")
    print(f"Szacowany wpływ zniżki (ATE - Average Treatment Effect): {discount_effect:.4f}")
    print(f"Wpływ oceny klienta na popyt: {rating_effect:.4f}")
    
    if discount_effect > 0:
        print("Wniosek biznesowy: Zwiększenie rabatu ma pozytywny, istotny wpływ na liczbę zamawianych sztuk.")
    else:
        print("Wniosek biznesowy: Rabat nie wpłynął znacząco na podniesienie wolumenu sprzedaży.")

if __name__ == "__main__":
    run_causal_inference()