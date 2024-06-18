# Importation des bibliothèques nécessaires
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO('C:/Users/deyha/Desktop/detection-vaisselle/runs/detect/train6/weights/best.pt')

# Charger une image à analyser
image_path = 'C:/Users/deyha/Desktop/detection-vaisselle/LIQUIDE-VAISSELLE-1/train/images/image_viber_2024-05-23_08-46-56-055_jpg.rf.5e1c13417962a2d7657f822fc5171c54.jpg'
image = cv2.imread(image_path)
results = model(image)
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        cls = box.cls[0]
        print(f'Objet détecté: {model.names[int(cls)]}')
        print(f'Coordonnées: ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})')
        print(f'Confiance: {confidence:.2f}')
        print('-' * 30)
    # Afficher les résultats avec OpenCV
    result.show()

    # Si vous souhaitez afficher l'image avec matplotlib
    result_image = result.plot()
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.show()