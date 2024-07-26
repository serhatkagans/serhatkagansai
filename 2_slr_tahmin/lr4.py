import streamlit as st
import pickle
import os

st.title('TV Reklam ve Satış Tahmin Uygulaması')

# Dosya yolunu dinamik olarak belirleyin
model_path = os.path.join(os.path.dirname(__file__), 'linear_regression_model.pkl')

# Modeli pickle dosyasından yükleyin
try:
    with open(model_path, 'rb') as file:
        lr = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model dosyası bulunamadı. Lütfen '{model_path}' dosyasının doğru konumda olduğundan emin olun.")
except Exception as e:
    st.error(f"Model dosyası yüklenirken bir hata oluştu: {e}")
else:
    # Slider ile TV reklam bütçesi girişi
    budget = st.slider('TV Reklam Bütçesi Girin:', min_value=0, max_value=300, value=100)
    try:
        prediction = lr.predict([[budget]])
        st.write(f"{budget} birim TV reklam bütçesi ile beklenen satış: {prediction[0][0]:.2f}")
    except Exception as e:
        st.error(f"Tahmin yapılırken bir hata oluştu: {e}")
