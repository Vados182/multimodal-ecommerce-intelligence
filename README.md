# Multimodal E-Commerce Intelligence Platform 🚀

[![CI/CD Pipeline](https://github.com/Vados182/multimodal-ecommerce-intelligence/actions/workflows/main.yml/badge.svg)](https://github.com/Vados182/multimodal-ecommerce-intelligence/actions)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

Produkcyjny, produkcyjnie zoptymalizowany system analityczny dla e-commerce łączący **Generatywną AI (LLM Agents)**, **Computer Vision (PyTorch)**, **Inżynierię Danych (PySpark ETL)** oraz **Wnioskowanie Przyczynowe (Causal Inference)**. 

Aplikacja została zaprojektowana z myślą o architekturze mikroserwisowej i zoptymalizowana pod kątem ścisłych ograniczeń pamięciowych środowisk chmurowych (np. Render 512MB RAM).

---

## 🏗️ Architektura Systemu

System działa w modelu dwuwarstwowym:
1. **Backend (FastAPI)** – Serwuje punkty końcowe REST API dla modeli ML/DL oraz Agenta LLM.
2. **Frontend (Streamlit)** – Interaktywny pulpit nawigacyjny dla kadr zarządczych i analityków e-commerce.

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw CSV Orders / Products] -->|PySpark ETL Pipeline| B[(Processed Parquet Storage)]
    end

    subgraph Backend Services - FastAPI
        B -->|Fetch Aggregated Data| C[Causal Inference Engine]
        D[Product Images] -->|Inference| E[MobileNetV3 PyTorch Classifier]
        F[Product Prompts] -->|Async Request| G[OpenAI Agent / Mock Fallback]
    end

    subgraph Application & User Interface
        C --> H[Streamlit Dashboard]
        E --> H
        G --> H
    end