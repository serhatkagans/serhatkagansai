import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('TV Reklam ve Satış Tahmin Uygulaması')

# Dosya yolunu belirleyin
file_path = os.path.join(os.path.dirname(__file__), 'reklam.csv')


data = pd.read_csv(file_path)

st.write("Veri Seti İlk 5 Satır")
st.write(data.head())


x = data.TV.values.reshape(-1, 1)
y = data.satış.values.reshape(-1, 1)


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)


lr = LinearRegression()
lr.fit(xtrain, ytrain)


yhead = lr.predict(xtest)


fig, ax = plt.subplots()
ax.scatter(x, y, label='Gerçek Veriler')
ax.plot(xtest, yhead, color='red', label='Model Tahminleri')
ax.set_xlabel("TV Reklam")
ax.set_ylabel("Satış")
ax.legend()
st.pyplot(fig)


budget = st.slider('TV Reklam Bütçesi Girin:', min_value=0, max_value=300, value
