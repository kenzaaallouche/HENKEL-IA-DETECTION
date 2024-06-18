# Importation des bibliothèques nécessaires
from ultralytics import YOLO
import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Configuration de l'application Streamlit
st.title("HENKEL Détection PRIL VS BINGO")
logo_path = 'Henkel.png'  # Remplacez par le chemin réel de votre logo

st.divider()
st.sidebar.image(logo_path,caption="HENKEL IA DETECTION ")


st.sidebar.header('Choisissez BRAND')
# Afficher le logo
# Model Options
confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
# Charger le modèle YOLOv8 pré-entraîné


# Télécharger une image à analyser
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image téléchargée
    res = model.predict(uploaded_file,
                        conf=confidence
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
    
