import tensorflow as tf                           # narzędzie do budowania sieci
from tensorflow.keras import layers, models       # Keras: warstwy i modele
import os                                         # obsługa ścieżek i plików
import matplotlib.pyplot as plt                   # tworzenie wykresów (loss/accuracy)
import numpy as np                                # operacje matematyczne na tablicach


# ścieżka do datasetu
# folder z danymi - wypakowany folder cats_and_dogs_filtered
base_dir = r"E:\Python pies kot\cats_and_dogs_filtered"

train_dir = os.path.join(base_dir, 'train')           # podfolder z danymi treningowymi
validation_dir = os.path.join(base_dir, 'validation') # podfolder z danymi walidacyjnymi

print("Train dir:", train_dir)
print("Validation dir:", validation_dir)

# parametry modelu
IMG_SIZE = (160, 160)  # docelowa wielkość obrazków (przeskalowane do 160x160)
BATCH_SIZE = 32        # liczba obrazów przetwarzanych naraz przez sieć (mini-batch)

# wczytanie datasetów
# image_dataset_from_directory:
# automatycznie tworzy dataset
# czyta obrazy z podfolderów (Cat, Dog)
# przydziela etykiety na podstawie nazw folderów
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,              # miesza dane - zapobiega overfittingowi
    batch_size=BATCH_SIZE,     # wielkość batcha
    image_size=IMG_SIZE        # resize obrazów do 160x160
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,              # miesza walidację
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# normalizacja zdjęć
# wartości pikseli: 0–255 -> normalizacja do zakresu 0-1
normalization_layer = layers.Rescaling(1. / 255)

# .map() stosuje funkcję do każdej paczki (batch)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# model CNN
# CNN – Convolutional Neural Network
model = models.Sequential([
    tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # wejście: obraz 160x160, 3 kanały (RGB)

    # blok 1
    layers.Conv2D(32, (3, 3), activation='relu'),     # filtry wykrywają krawędzie/tekstury
    layers.MaxPooling2D(2, 2),                        # zmniejsza rozdzielczość 2x, zachowując cechy

    # blok 2
    layers.Conv2D(64, (3, 3), activation='relu'),     # głębsze filtry — wykrywanie bardziej złożonych wzorców
    layers.MaxPooling2D(2, 2),

    # blok 3
    layers.Conv2D(128, (3, 3), activation='relu'),    # więcej filtrów — kształty, fragmenty obiektów
    layers.MaxPooling2D(2, 2),

    # wyjście cnn
    layers.Flatten(),                                 # zamiana map cech 2D na jedną długą listę wartości

    # warstwy gęste
    layers.Dense(128, activation='relu'),             # 128 neuronów - łączy wszystkie cechy
    layers.Dense(1, activation='sigmoid')             # Wyjście 0–1 -> prawdopodobieństwo (pies/kot)
])


# kompilacja modelu
# optimizer='adam' - uczy się efektywnie
# loss='binary_crossentropy' - idealny dla klasyfikacji binarnej
# metrics=['accuracy'] - monitoruje dokładność
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# trening modelu
print("Trenuję model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=5)         # fit() uruchamia proces uczenia na zbiorze treningowym


# ewaluacja
loss, acc = model.evaluate(val_ds)  # Sprawdzenie modelu na danych spoza treningowego zbioru
print("Dokładność na zbiorze walidacyjnym:", acc)

# wizualizacja historii treningów
acc = history.history['accuracy']          # dokładność na treningu
val_acc = history.history['val_accuracy']  # dokładność na walidacji
loss = history.history['loss']             # strata na treningu
val_loss = history.history['val_loss']     # strata na walidacji

epochs_range = range(len(acc))             # zakres epok

plt.figure(figsize=(12, 4))                # ustawia rozmiar wykresów

# wykres dokładności 
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# wkyres straty 
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# zapis modelu
model.save("cats_vs_dogs_model.h5")        # zapis pełnego modelu (architektura + wagi)
print("Model zapisany jako cats_vs_dogs_model.h5")


# predykcja na nowym własnym zdjęciu
img_path = r"E:\Python pies kot\pies.jpg"  # ścieżka do zdjęcia testowego

if os.path.exists(img_path):               # jeśli plik istnieje -> wczytaj
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)       # ładuje i zmienia rozmiar

    img_array = tf.keras.utils.img_to_array(img)                        # zamienia obraz do tablicy liczb (pikseli)

    img_array = tf.expand_dims(img_array, 0) / 255.0                    # dodaje wymiar batch (model wymaga kształtu [1, 160, 160, 3])

    prediction = model.predict(img_array)  # model zwraca wartość 0–1 - prawdopodobieństwo

    if prediction[0][0] > 0.5:             # > 0.5 - pies
        print("Wygląda na psa ")
    else:                                  # < 0.5 - kot
        print("Wygląda na kota ")

else:
    print("Nie znaleziono pliku w folderze.")