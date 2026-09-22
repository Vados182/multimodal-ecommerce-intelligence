import os
import pandas as pd
import numpy as np

def generate_ecommerce_data(output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    n_products = 500
    n_orders = 10000

    categories = ['Electronics', 'Home & Kitchen', 'Automotive', 'Fashion', 'Industrial']

    products_data = {
        'product_id': [f"PRD-{i:04d}" for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, size=n_products),
        'base_price': np.round(np.random.uniform(10.0, 500.0, size=n_products), 2),
        'raw_description': [
            f"Super okazja! Produkt PRD-{i:04d}. Stan idealny, szybka wysyłka. Idealne do domowego użytku."
            for i in range(1, n_products + 1)
        ],
        'image_path': [f"data/raw/images/img_{i:04d}.jpg" for i in range(1, n_products + 1)]
    }
    df_products = pd.DataFrame(products_data)
    df_products.to_csv(os.path.join(output_dir, "products.csv"), index=False)

    dates = pd.date_range(start="2026-01-01", end="2026-09-01", freq="h")
    orders_data = {
        'order_id': [f"ORD-{i:07d}" for i in range(1, n_orders + 1)],
        'product_id': np.random.choice(df_products['product_id'], size=n_orders),
        'timestamp': np.random.choice(dates, size=n_orders),
        'quantity': np.random.randint(1, 5, size=n_orders),
        'discount_applied': np.random.choice([0.0, 0.05, 0.1, 0.2], size=n_orders, p=[0.6, 0.2, 0.1, 0.1]),
        'customer_rating': np.random.choice([1, 2, 3, 4, 5], size=n_orders, p=[0.05, 0.05, 0.1, 0.3, 0.5])
    }
    df_orders = pd.DataFrame(orders_data)
    df_orders.to_csv(os.path.join(output_dir, "orders.csv"), index=False)

    print(f"Wygenerowano dane w {output_dir}:")
    print(f"- products.csv ({len(df_products)} wierszy)")
    print(f"- orders.csv ({len(df_orders)} wierszy)")

if __name__ == "__main__":
    generate_ecommerce_data()