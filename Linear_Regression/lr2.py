import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('TV Reklam ve Satış Tahmin Uygulaması')

# CSV dosyasını yükleyin
data = pd.read_csv("reklam.csv")

st.write("Veri Seti İlk 5 Satır")
st.write(data.head())

# Veriyi hazırlayın
x = data.TV.values.reshape(-1, 1)
y = data.satış.values.reshape(-1, 1)

# Veriyi eğitim ve test olarak ayırın
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

# Modeli eğitin
lr = LinearRegression()
lr.fit(xtrain, ytrain)

# Tahmin yapın
yhead = lr.predict(xtest)

# Scatter plot ve tahmin çizgisi oluşturun
fig, ax = plt.subplots()
ax.scatter(x, y, label='Gerçek Veriler')
ax.plot(xtest, yhead, color='red', label='Model Tahminleri')
ax.set_xlabel("TV Reklam")
ax.set_ylabel("Satış")
ax.legend()
st.pyplot(fig)

# Slider ile TV reklam bütçesi girişi
budget = st.slider('TV Reklam Bütçesi Girin:', min_value=0, max_value=300, value=100)
prediction = lr.predict([[budget]])
st.write(f"{budget} birim TV reklam bütçesi ile beklenen satış: {prediction[0][0]:.2f}")
