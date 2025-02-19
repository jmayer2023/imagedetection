# Importieren der benötigten Bibliotheken

# Verarbeitung von Dateipfaden auf unterschiedlichen Betriebssystemen
import os
# Zahlenerweiterung zur Übergabe des Datenarrays
import numpy as np
# KI Modell um Bilder zu lesen und auszugeben
import cv2 as cv
# KI Modell um die Daten zu analysieren
import tensorflow as tf

# Modell laden
model = tf.keras.models.load_model('insektenstich_model.h5')

# Pfad zum Bild
PATH_TO_TRAINING_IMAGES_FOLDER = "images/training"
insects = os.listdir(PATH_TO_TRAINING_IMAGES_FOLDER)

# zu prüfendes Bild einlesen
img = cv.imread("images/validation/feuerameise.jpg")

# Bildvorverarbeitung
# Größe anpassen
img_resized = cv.resize(img, (150, 150))  
# anpassen der Daten für die KI (Normalisierung)
img_normalized = img_resized.astype('float32') / 255.0 
# hinzufügen einer X-Achse (Dimension) -> 3D -> 4D, damit das Format dem KI-Modell entspricht 
img_expanded = np.expand_dims(img_normalized, axis=0)

# Vorhersage mit dem Keras-Modell
predictions = model.predict(img_expanded)
# Index der Klasse (Bild/Insekt) mit der höchsten Wahrscheinlichkeit
label = np.argmax(predictions[0])
# Vorhersage-Wahrscheinlichkeit
confidence = predictions[0][label] 

# Ausgabe der Vorhersage in der Konsole
print(f'Vorhergesagte Klasse: {insects[label]}, Vertrauen (1 = max): {confidence:.2f}')

# Ausgabe

# Füge Text dem Bild hinzu
cv.putText(img, f"Insekt: {insects[label]} Genauigkeit (1 = max): ({confidence:.2f})", (10, 30), cv.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)
# Zeige Bild an
cv.imshow("Vorhersage", img)
# Zeige an bis geschlossen
cv.waitKey(0)
# Speicher löschen
cv.destroyAllWindows()