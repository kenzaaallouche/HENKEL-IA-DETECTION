# Importation des bibliothèques nécessaires
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Chemin vers le fichier de poids du modèle
model_path = 'best.pt'

# Configuration de la page
st.set_page_config(
    page_title="Détection d'objets",  # Titre de la page
    page_icon=":robot_face:",  # Icône de la page
    layout="wide",  # Mise en page large
    initial_sidebar_state="expanded"  # Barre latérale initialement étendue
)

# Barre latérale pour le téléchargement de l'image et la configuration du modèle
with st.sidebar:
    st.header("Configuration de l'image")
    source_img = st.file_uploader("Télécharger une image...", type=["jpg", "jpeg", "png"])

    confidence = st.slider("Confiance du modèle", 0.25, 1.0, 0.4, step=0.05)

# Titre principal
st.title("Détection d'objets")

# Affichage de l'image téléchargée si elle existe
if source_img is not None:
    uploaded_image = Image.open(source_img)
    st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)

# Chargement du modèle YOLO
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Impossible de charger le modèle. Veuillez vérifier le chemin spécifié: {model_path}")
    st.error(ex)

# Bouton pour détecter les objets
if st.sidebar.button('Détecter les objets'):
    if source_img is not None:
        # Prédiction des objets dans l'image téléchargée
        results = model(source_img, conf=confidence)

        # Affichage de l'image avec les boîtes englobantes
        st.image(results.render()[0], caption='Image avec détection', use_column_width=True)

        # Affichage des résultats de détection
        with st.expander("Résultats de la détection"):
            for box in results.pandas().xywh[0].iterrows():
                st.write(box[1])
    else:
        st.warning("Veuillez d'abord télécharger une image.")
