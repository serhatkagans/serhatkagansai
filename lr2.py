import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("reklam.csv")
data.head()

x=data.TV.values.reshape(-1,1)
y=data.satış.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("TV Reklam")
plt.ylabel("Satış")
plt.show()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(xtrain,ytrain)

yhead=lr.predict(xtest)

plt.scatter(x,y)
plt.plot(xtest,yhead)

lr.predict([[100]])

#%%
import streamlit as st
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(xtest, yhead, color='red')
ax.set_xlabel("TV Reklam")
ax.set_ylabel("Satış")
st.pyplot(fig)

#%%


budget = st.slider('TV Reklam Bütçesi Girin:', min_value=0, max_value=300, value=100)
prediction = lr.predict([[budget]])
st.write(f"{budget} birim TV reklam bütçesi ile beklenen satış: {prediction[0][0]:.2f}")