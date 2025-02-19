# Importieren der benötigten Bibliotheken

# Verarbeitung von Dateipfaden auf unterschiedlichen Betriebssystemen
import os
# Zahlenerweiterung zur Übergabe des Datenarrays
import numpy as np
# KI Modell um Bilder zu lesen und auszugeben
import cv2
# KI Modell um die Daten zu analysieren
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Parameter vorbereiten

# KI layer und model
layers = tf.keras.layers
models = tf.keras.models
# Festlegung der Bildgröße
IMG_HEIGHT, IMG_WIDTH = 150, 150
# Anzahl der gleichzeitig verarbeiteten Bilder bevor Gewichtung aktualisiert (32 Kompromiss zwischen Speicherverbrauch und Trainingseffizienz) [kein Einfluss auf "test accuracy"]  
BATCH_SIZE = 32
# Durchläufe der Trainingsdaten (ab 24 "test accuracy: 0.00")
EPOCHS = 23

# Funktion zum Laden der Daten
def load_data(data_dir):
    # Leere Listen
    images = []
    labels = []
    # Bezeichnung der "Bilder/Insekten"
    class_names = os.listdir(data_dir)

    # Laden der einzelnen Bilder
    for class_index, class_name in enumerate(class_names):
        # Erstellen des Pfades
        class_dir = os.path.join(data_dir, class_name)

        # aktuelles Bild
        for img_name in os.listdir(class_dir):
            # erstelle Pfad zum Bild
            img_path = os.path.join(class_dir, img_name)
            # lese Bild
            img = cv2.imread(img_path)
            # passe Größe an
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) 
            # füge es den Listen hinzu 
            images.append(img)
            # Klasse als Label
            labels.append(class_index) 

    # Rückgabe der Daten als Numpy-Array
    return np.array(images), np.array(labels), class_names

# Magic

# einlesen der Daten aus dem Trainingsordner
data_dir = "images/training"  
# speichern der Daten aus der Funktion in Variablen
X, y, class_names = load_data(data_dir)
# anpassen der Daten für die KI
X = X.astype('float32') / 255.0
# Aufteilen in Trainings- und Testdatensatz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KI Modell erstellen
model = models.Sequential([
    # Convoluional-Schicht, 32 Filter, 3x3 Kernel, ReLU-Aktivierungsfunktion, Form des Bildes mit Anzahl Farbkanäle (3)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # Pooling: reduziert die Dimensionen der Merkmalskarten auf die Hälfte
    layers.MaxPooling2D(pool_size=(2, 2)),
    # nächste Convoluional-Schicht mit 64 Filtern
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Pooling
    layers.MaxPooling2D(pool_size=(2, 2)),
    # nächste Convoluional-Schicht mit 128 Filtern
    layers.Conv2D(128, (3, 3), activation='relu'),
    # Pooling
    layers.MaxPooling2D(pool_size=(2, 2)),
    # wandelt Merkmalskanten in Vektoren um
    layers.Flatten(),
    # Dense-Schicht mit 128 Neuronen
    layers.Dense(128, activation='relu'),
    # Ausgangsschicht: Anzahl "klassen" = Anzahl Neuronen, Softmax-Aktivierungsfunktion
    layers.Dense(len(class_names), activation='softmax')
])

# Modell kompilieren
model.compile(
            # KI Optimieremodell
            optimizer='adam',
            # Verlustfunktion (zwischen vorhergesagter Wahrscheinlichkeit und tatsächlichen label)
            loss='sparse_categorical_crossentropy',
            # Metrik zur Bewertung der Genauigkeit
            metrics=['accuracy'])

# Modell trainieren
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Modell speichern
model.save('insektenstich_model.h5')

# Modell evaluieren
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.2f}')