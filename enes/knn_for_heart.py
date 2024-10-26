import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st 

class KNNModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.x = self.data.iloc[:, 1:-1].values
        self.y = self.data.iloc[:, -1].values
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, test_size=0.2, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)
        self.x_train = self.scaler.fit_transform(x_train)
        self.x_test = self.scaler.transform(x_test)
        self.y_train = y_train
        self.y_test = y_test

    def train(self, n_neighbors):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)
    
    def accuracy(self):
        y_pred = self.predict()
        return accuracy_score(self.y_test, y_pred)
    
    def confusion_matrix(self):
        y_pred = self.predict()
        return confusion_matrix(self.y_test, y_pred)
    
def manual_input_features():
    age = st.slider("Yaş", 20, 80, 50)
    sex = st.slider("Cinsiyet (1: Erkek, 0: Kadın)", 0, 1, 1)
    cp = st.slider("Göğüs Ağrısı Tipi (0-3)", 0, 3, 1)
    trestbps = st.slider("Kan Basıncı", 90, 200, 120)
    chol = st.slider("Kolesterol", 100, 400, 200)
    fbs = st.slider("Açlık Kan Şekeri (1: Evet, 0: Hayır)", 0, 1, 0)
    restecg = st.slider("EKG Sonucu (0-2)", 0, 2, 1)
    thalach = st.slider("Max Kalp Atış Sayısı", 70, 210, 150)
    exang = st.slider("Egzersizde Anjina (1: Evet, 0: Hayır)", 0, 1, 0)
    oldpeak = st.slider("ST Depresyonu", 0.0, 6.0, 2.0)
    slope = st.slider("ST Eğimi (0-2)", 0, 2, 1)
    ca = st.slider("Renklenmiş Damar Sayısı (0-4)", 0, 4, 0)
    thal = st.slider("Thal (0-3)", 0, 3, 2)

    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(-1, 1)
    
    if features.shape[1] != 13:
        st.error("Girdi özellikleri hatalı! Lütfen tüm özellikleri doğru giriniz.")
    
    return features

def main():
    st.title("KNN Sınıflandırıcı")

    knn_model = KNNModel("heart.csv")
    knn_model.prepare_data()

    n_neighbors = st.slider("n_neighbors Değeri", min_value=1, max_value=20, value=5)

    knn_model.train(n_neighbors)

    accuracy = knn_model.accuracy()
    st.write(f"Doğru Tahmin Oranı (Accuracy): {accuracy:.2f}")

    predictions = knn_model.predict()
    prediction_df = pd.DataFrame({
        'Gerçek Değerler': knn_model.y_test,
        'Tahmin Değerleri': predictions
    })
    st.write(prediction_df)

    st.write("Manuel Veri Girişi")
    user_input = manual_input_features()

    scaled_user_input = knn_model.scaler.transform(user_input) 

    user_prediction = knn_model.model.predict(scaled_user_input)
    st.write(f"Verilen veriler için tahmin edilen grup: {user_prediction[0]}")

if __name__ == "__main__":
    main()
    
