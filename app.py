import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('/content/drive/MyDrive/CURSO DE IA Y BIG DATA/PIA/REDES NEURONALES/models/fashion_mnist.keras')

# Crear la interfaz de usuario
st.title("Clasificador Fashion MNIST")
st.write("Sube una imagen para clasificarla como una categoría de ropa.")

# Subir la imagen
uploaded_file = st.file_uploader("Sube una imagen en escala de grises de 28x28 píxeles.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L') # Convertir RGB a blanco y negro
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0 # Normalizar
    # El primer 1 indica que sólo hay una iamgen, luego las dimensiones
    # y el último 1 indica que sólo hay un canal de color
    img_array = img_array.reshape(1, 28, 28, 1)

  # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Realizar la predicción
    prediction = model.predict(img_array)
    classes = ["Camiseta/Top", "Pantalón", "Jersey", "Vestido", "Abrigo", "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]
    st.write("Predicción:", classes[np.argmax(prediction)])
