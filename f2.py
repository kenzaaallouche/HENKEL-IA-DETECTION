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

# Charger le modèle YOLOv8 pré-entraîné

#model = YOLO('./runs/detect/train6/weights/best.pt')
try:
    model = YOLO('best.pt')
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: best.pt")
    st.error(ex)
# Télécharger une image à analyser
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image téléchargée
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convertir l'image en BGR pour OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Effectuer la détection des objets
    results = model(image_bgr)

    # Dictionnaire pour compter le nombre de chaque produit détecté
    product_counts = {}

    # Compter le nombre total de produits détectés
    total_detected = 0
    for result in results:
        boxes = result.boxes
        total_detected += len(boxes)
        for box in boxes:
            cls = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[cls]
            
            if class_name not in product_counts:
                product_counts[class_name] = 0
            product_counts[class_name] += 1

            # Afficher les résultats avec matplotlib
            

    # Afficher le nombre total de produits détectés
    st.header('Nombre total de produits détectés:')
    st.write(total_detected)

    # Afficher le nombre de produits détectés par classe avec barres de progression
    st.header('Nombre de produits détectés par classe:')
    for product, count in product_counts.items():
        percentage = count / total_detected * 100
        st.write(f'{product}: {count} ({percentage:.2f}%)')
        st.progress(percentage / 100.0)
    result_image = result.plot()
    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption='Image avec détections', use_column_width=True)
