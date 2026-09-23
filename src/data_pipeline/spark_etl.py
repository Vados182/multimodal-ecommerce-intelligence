import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, sum, avg, count

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
    print("Dane syntetyczne wygenerowane w data/raw/")

def run_spark_etl(input_dir="data/raw", output_dir="data/processed"):
    spark = SparkSession.builder \
        .appName("ECommerceDataETL") \
        .master("local[*]") \
        .getOrCreate()

    print("Uruchamianie ETL w PySpark...")

    products_df = spark.read.csv(os.path.join(input_dir, "products.csv"), header=True, inferSchema=True)
    orders_df = spark.read.csv(os.path.join(input_dir, "orders.csv"), header=True, inferSchema=True)

    joined_df = orders_df.join(products_df, on="product_id", how="inner")
    
    enriched_df = joined_df.withColumn(
        "final_price", round(col("base_price") * (1 - col("discount_applied")), 2)
    ).withColumn(
        "total_revenue", round(col("final_price") * col("quantity"), 2)
    )

    category_kpi = enriched_df.groupBy("category").agg(
        sum("total_revenue").alias("total_category_revenue"),
        avg("customer_rating").alias("avg_rating"),
        count("order_id").alias("total_orders")
    )

    print("\n Podsumowanie sprzedaży per kategoria:")
    category_kpi.show()

    os.makedirs(output_dir, exist_ok=True)
    enriched_df.toPandas().to_parquet(os.path.join(output_dir, "enriched_orders.parquet"), index=False)
    
    print(f"Dane przetworzone i zapisane w formatach Parquet w folderze {output_dir}")
    spark.stop()

if __name__ == "__main__":
    generate_ecommerce_data()
    run_spark_etl()