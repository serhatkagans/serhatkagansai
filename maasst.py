
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("reklam.csv")
data.head()


x=data.iloc[:,1:-1].values

y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=22)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)

yhead=lr.predict(xtest)
yhead

lr.predict([[44,39,45]])
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

rmse =np.sqrt(mean_squared_error(ytest,yhead))

r2=r2_score(ytest,yhead)

print("rmse değeri: ",rmse)
print("r2 değeri: ", r2)


# In[ ]:
import streamlit as st
st.title('Reklam Verisi ile Doğrusal Regresyon Modeli')
st.write("Veri Başlıkları:")
st.write(data.head())  # Verinin ilk birkaç satırını göster

tv_input = st.number_input("TV Reklam Harcaması", min_value=0.0, value=44.0)
radio_input = st.number_input("Radyo Reklam Harcaması", min_value=0.0, value=39.0)
newspaper_input = st.number_input("Gazete Reklam Harcaması", min_value=0.0, value=45.0)

# Kullanıcı girdilerini birleştir
user_values = np.array([[tv_input, radio_input, newspaper_input]])

# Tahmin yapma
predicted_value = lr.predict(user_values)

# Tahmin sonuçlarını göster
st.write(f"Verilen değerler için tahmin edilen sonuç: {predicted_value[0]}")