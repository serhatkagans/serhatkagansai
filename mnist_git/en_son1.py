import joblib
# Modeli tekrar yükleme
lr_loaded = joblib.load('logistic_regression_mnist_model.pkl')
print("Model yüklendi.")

# Yeni bir resimle tahmin yapma
from PIL import Image
import numpy as np

# Yeni bir resim açma, boyutlandırma ve gri tonlamaya çevirme
img = Image.open("2.png")  # Resim yolunu güncelleyin
img = img.resize((28, 28))
img = img.convert("L")

# Resmi numpy dizisine çevirme ve modelin kabul edeceği boyuta getirme
img_array = np.array(img).reshape(1, -1)  # 1 satırlık veriye çeviriyoruz

# Modelle tahmin yapma
pred = lr_loaded.predict(img_array)
print("Tahmin edilen sınıf:", pred)

# Tahmin edilen rakamı görselleştirme
import matplotlib.pyplot as plt
plt.imshow(img_array.reshape(28, 28), cmap="gray")
plt.title(f"Tahmin edilen rakam {pred[0]}")
plt.show()
#%%
import streamlit as st
st.title("MNIST Rakam Tanıma Uygulaması")


uploaded_file=st.file_uploader("lütfen resim yükleyin",type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img=Image.open(uploaded_file)
    img=img.resize((28,28))
    img=img.convert("L")
    
    img_array=np.array(img).reshape(1,-1)
    
    pred=lr_loaded.predict(img_array)
    st.write(f"tahmin edilen sınıf: {pred[0]}")
    
    st.image(img,caption=f"tahmin edilen rakam: {pred[0]}")