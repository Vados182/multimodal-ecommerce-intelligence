# Multimodal E-Commerce Intelligence Platform 🚀

[![CI/CD Pipeline](https://github.com/Vados182/multimodal-ecommerce-intelligence/actions/workflows/main.yml/badge.svg)](https://github.com/Vados182/multimodal-ecommerce-intelligence/actions)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

A production-ready, resource-optimized analytical platform for e-commerce combining **Generative AI (LLM Agents)**, **Computer Vision (PyTorch)**, **Data Engineering (PySpark ETL)**, and **Causal Inference**.

The application is engineered around a microservice architecture and optimized for strict memory constraints in cloud deployment environments (e.g., Render 512MB RAM).

---

## 🏗️ System Architecture

The system operates on a two-tier architecture:
1. **Backend (FastAPI)** – Serves REST API endpoints for ML/DL models and the LLM Agent.
2. **Frontend (Streamlit)** – Provides an interactive dashboard for executives and e-commerce analysts.

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