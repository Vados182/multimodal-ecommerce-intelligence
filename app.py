import streamlit as st
import requests

# Konfiguracja strony Streamlit
st.set_page_config(
    page_title="Multimodal E-Commerce Intelligence",
    page_icon="🛍️",
    layout="wide"
)

# Adres URL Twojego API wysłanego na Render
API_BASE_URL = "https://multimodal-ecommerce-intelligence.onrender.com"

st.title("🛍️ Multimodal E-Commerce Intelligence System")
st.markdown("Interaktywny panel analityczny łączący **GenAI Agent**, **Computer Vision (PyTorch)** oraz **Causal Inference**.")

# Zakładki interfejsu
tab1, tab2, tab3 = st.tabs([
    "🤖 Analiza Produktu (Agent GenAI)", 
    "🖼️ Rozpoznawanie Obrazu (Computer Vision)", 
    "📈 Wnioskowanie Przyczynowe (Causal Impact)"
])

# --- TAB 1: AGENT GENAI ---
with tab1:
    st.header("Analiza oferty e-commerce")
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input("Nazwa produktu", value="Słuchawki Sony WH-1000XM5")
        description = st.text_area("Opis lub zapytanie użytkownika", value="Czy warto kupić je do pracy w głośnym biurze?")
        submit_btn = st.button("Analizuj produkt", type="primary")

    with col2:
        if submit_btn:
            with st.spinner("Agent analizuje ofertę..."):
                try:
                    payload = {
                        "product_name": product_name,
                        "description": description
                    }
                    response = requests.post(f"{API_BASE_URL}/analyze-product", json=payload)
                    
                    if response.status_code == 200:
                        st.success("Analiza zakończona sukcesem!")
                        st.json(response.json())
                    else:
                        st.error(f"Błąd API: status {response.status_code}")
                except Exception as e:
                    st.error(f"Nie udało się połączyć z API: {e}")

# --- TAB 2: COMPUTER VISION ---
with tab2:
    st.header("Klasyfikacja zdjęć produktów (ResNet-18)")
    uploaded_file = st.file_uploader("Wgraj zdjęcie produktu", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col_img, col_res = st.columns(2)
        
        with col_img:
            st.image(uploaded_file, caption="Wgrany obraz", use_container_width=True)
            
        with col_res:
            if st.button("Uruchom model detekcji", type="primary"):
                with st.spinner("Model ResNet-18 przetwarza obraz..."):
                    try:
                        # Zamiast starych linii z dictem:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/analyze-image", files=files)
                                                
                        if response.status_code == 200:
                            st.success("Detekcja zakończona!")
                            st.json(response.json())
                        else:
                            st.error(f"Błąd API: status {response.status_code}")
                    except Exception as e:
                        st.error(f"Nie udało się połączyć z API: {e}")

# --- TAB 3: CAUSAL INFERENCE ---
with tab3:
    st.header("Predykcja wpływu zmian (Causal Impact)")
    st.info("Moduł pozwalający oszacować wpływ obniżek cen lub kampanii promocyjnych na konwersję.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        price_change = st.slider("Zmiana ceny (%)", min_value=-50, max_value=50, value=-10)
        ad_spend = st.number_input("Dodatkowy budżet reklamowy (PLN)", value=500)
        causal_btn = st.button("Oblicz wpływ", type="primary")
        
    with col_c2:
        if causal_btn:
            with st.spinner("Obliczanie estymacji..."):
                try:
                    payload = {"price_change_pct": price_change, "ad_spend": ad_spend}
                    response = requests.post(f"{API_BASE_URL}/predict-causal-impact", json=payload)
                    if response.status_code == 200:
                        st.success("Obliczenia wykonane!")
                        st.json(response.json())
                    else:
                        st.error(f"Błąd API: status {response.status_code}")
                except Exception as e:
                    st.error(f"Nie udało się połączyć z API: {e}")