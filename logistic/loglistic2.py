import streamlit as st
import pickle
import pandas as pd
import os

# Başlık
st.title("Göğüs Kanseri Teşhis Uygulaması")

# Dosya yollarını dinamik olarak oluştur
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Model ve scaler dosyalarını yükle
with open(scaler_path, "rb") as file:
    loaded_scaler = pickle.load(file)

with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)

# Kullanıcıdan manuel olarak özellik değerlerini al
st.header("Özellik Değerlerini Giriniz:")

def user_input_features():
    radius_mean = st.number_input("Radius Mean", value=17.99)
    texture_mean = st.number_input("Texture Mean", value=10.38)
    perimeter_mean = st.number_input("Perimeter Mean", value=122.8)
    area_mean = st.number_input("Area Mean", value=1001.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.1184)
    compactness_mean = st.number_input("Compactness Mean", value=0.2776)
    concavity_mean = st.number_input("Concavity Mean", value=0.3001)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.1471)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.2419)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.07871)
    radius_se = st.number_input("Radius SE", value=1.095)
    texture_se = st.number_input("Texture SE", value=0.9053)
    perimeter_se = st.number_input("Perimeter SE", value=8.589)
    area_se = st.number_input("Area SE", value=153.4)
    smoothness_se = st.number_input("Smoothness SE", value=0.006399)
    compactness_se = st.number_input("Compactness SE", value=0.04904)
    concavity_se = st.number_input("Concavity SE", value=0.05373)
    concave_points_se = st.number_input("Concave Points SE", value=0.01587)
    symmetry_se = st.number_input("Symmetry SE", value=0.03003)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.006193)
    radius_worst = st.number_input("Radius Worst", value=25.38)
    texture_worst = st.number_input("Texture Worst", value=17.33)
    perimeter_worst = st.number_input("Perimeter Worst", value=184.6)
    area_worst = st.number_input("Area Worst", value=2019.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.1622)
    compactness_worst = st.number_input("Compactness Worst", value=0.6656)
    concavity_worst = st.number_input("Concavity Worst", value=0.7119)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.2654)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.4601)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.1189)
    
    data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Özellikleri ölçeklendir
ozellikler_olceklendirilmis = loaded_scaler.transform(input_df)

# Tahmin yap
tahmin = loaded_model.predict(ozellikler_olceklendirilmis)

# Sonuçları göster
st.subheader('Tahmin Edilen Teşhis Durumu:')
st.write('Kötü Huylu' if tahmin[0] == 1 else 'İyi Huylu')

# SHAP değerlerini hesapla ve göster
explainer = shap.Explainer(loaded_model)
shap_values = explainer(ozellikler_olceklendirilmis)

st.header("SHAP Değerleri")
shap.summary_plot(shap_values, input_df, plot_type="bar")
