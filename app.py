import streamlit as st
import joblib
import numpy as np

# Judul Aplikasi
st.title("🏠 Prediksi Harga Rumah")
st.write("""
Aplikasi ini memprediksi harga rumah berdasarkan fitur-fitur seperti:
jumlah kamar, luas bangunan, tahun dibangun, dll.
""")

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load('svr_house_price.pkl')
    sc_X = joblib.load('feature_scaler.pkl')
    sc_Y = joblib.load('target_scaler.pkl')
    return model, sc_X, sc_Y

model, sc_X, sc_Y = load_model()

# Input Fitur
st.sidebar.header("Masukkan Fitur Rumah")

bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", 1, 10, 3)
bathrooms = st.sidebar.slider("Jumlah Kamar Mandi", 1, 5, 2)
sqft_living = st.sidebar.number_input("Luas Bangunan (sqft)", 500, 10000, 1800)
sqft_lot = st.sidebar.number_input("Luas Tanah (sqft)", 500, 50000, 5000)
floors = st.sidebar.slider("Jumlah Lantai", 1, 4, 1)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1], 0)
view = st.sidebar.slider("View Quality (0-4)", 0, 4, 0)
condition = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
yr_built = st.sidebar.number_input("Tahun Dibangun", 1900, 2025, 1995)

# Tombol Prediksi
if st.sidebar.button("Prediksi Harga"):
    # Format input
    input_features = [
        bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
        view, condition, grade, sqft_living, sqft_lot, yr_built, 0, 47.6, -122.3, sqft_living, sqft_lot
    ]
    
    # Scaling dan prediksi
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = sc_X.transform(input_array)
    pred_scaled = model.predict(input_scaled)
    pred_price = sc_Y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    # Tampilkan hasil
    st.success(f"### Prediksi Harga: **${pred_price[0][0]:,.2f}**")
    st.balloons()

# Catatan
st.markdown("---")
st.caption("""
*Catatan:* Model ini menggunakan SVR yang telah dilatih dengan data historis.
""")