import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import wget
import zipfile
import os
import contextlib

# Título de la aplicación
st.title("Detección de Neumonía en Imágenes de Rayos X")
st.write("Utiliza la inteligencia artificial para analizar imágenes de rayos X y detectar signos de neumonía.")
st.write(f"**Versión de TensorFlow:** {tf.__version__}")

# Descargar y descomprimir el modelo si no existe
def download_and_extract_model():
    model_url = 'https://dl.dropboxusercontent.com/s/vkjkcpl5gk0p5gg9peohi/best_model.zip?rlkey=p6z4sbxs4na8jum0la0da2d1s&st=tays1yit'
    zip_path = 'best_model.zip'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return False

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    return os.path.join(extract_folder, 'best_model_local.keras')

modelo_path = download_and_extract_model()

# Verificar si el archivo del modelo existe
if not modelo_path or not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo.")
else:
    st.success("Archivo del modelo encontrado.")

# Definir el modelo base InceptionV3
base_model = InceptionV3(weights=None, include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# Añadir capas de clasificación
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Cargar los pesos del modelo desde el archivo .keras
try:
    model.load_weights(modelo_path)
    st.success("Pesos del modelo cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")

# Verificación de carga de archivo
st.markdown("### ¡Sube una imagen de rayos X!")
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar la imagen

    # Realizar la predicción
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('🔴 El modelo predice que la imagen muestra **signos de neumonía**.')
        st.markdown("### Recomendación:")
        st.write("Se sugiere consultar a un médico para un diagnóstico más detallado.")
    else:
        st.success('🟢 El modelo predice que la imagen **no muestra signos de neumonía**.')
        st.markdown("### Nota:")
        st.write("Si tienes dudas o síntomas, es recomendable consultar a un profesional de la salud.")
