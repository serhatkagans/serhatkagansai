import os
import joblib
import PIL.Image as img
import numpy as np
# Modeli yükleme

model_path = os.path.join(os.path.dirname(__file__), 'covid_model.pkl')
clf_loaded = joblib.load(model_path)
print("Model yüklendi.")

# Yeni bir resimle tahmin yapma
"""test = "deneme/resim1.png"
goruntu_oku1 = img.open(test).convert("L")
goruntu_boyutlandirma1 = goruntu_oku1.resize((28, 28))
goruntu_donusturme1 = np.array(goruntu_boyutlandirma1).flatten()
goruntu_donusturme1 = np.reshape(goruntu_donusturme1, (1, -1))

tahmin = clf_loaded.predict(goruntu_donusturme1)

print("Tahmin edilen sınıf:", tahmin)

if tahmin == [0]:
    print("covidli")
else:
    print("covid değil")
"""
#%%
import streamlit as st
st.title("COVID Görüntü Tanıma Uygulaması")

uploaded_file = st.file_uploader("Lütfen bir COVID görüntüsü yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Yüklenen resmi açma ve işleme
    image = img.open(uploaded_file).convert("L")  # Gri tonlamaya çeviriyoruz
    image = image.resize((28, 28))  # 28x28 boyutlandırma
    img_array = np.array(image).flatten()  # Resmi düzleştirme
    img_array = np.reshape(img_array, (1, -1))  # Modelin beklediği boyuta getiriyoruz
    
    # Modelle tahmin yapma
    tahmin = clf_loaded.predict(img_array)
    
    # Sonuçları yazdırma
    st.image(image, caption='Yüklenen Resim')
    
    if tahmin == [0]:
        st.write("Tahmin edilen sınıf: COVID'li")
    else:
        st.write("Tahmin edilen sınıf: COVID değil")
