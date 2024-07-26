import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Modeli yükle
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Manuel resimle test etme
img = Image.open("2.png")
img = img.resize((28,28))
img = img.convert("L")
img_array = np.array(img).reshape(1, -1)

# Tahmin yap
pred = loaded_model.predict(img_array)
print(f"Tahmin edilen rakam: {pred}")

# İsteğe bağlı olarak görüntüyü gösterme
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Tahmin edilen rakam: {pred[0]}")
plt.show()
