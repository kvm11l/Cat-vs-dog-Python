# Kot vs Pies

## Opis projektu
Prosty projekt klasyfikacji obrazów wykorzystujący sieć neuronową CNN (Convolutional Neural Network) do rozpoznawania, czy na zdjęciu znajduje się pies czy kot. \
Napisany w Pythonie z użyciem TensorFlow/Keras i pokazuje kompletny pipeline uczenia maszynowego: \
od przygotowania danych -> trenowania -> oceny -> zapisu modelu -> predykcji na własnym zdjęciu. \
Celem było zrozumienie podstawy uczenia maszynowego i przetwarzania obrazów oraz własna ciekawość.

## Działanie
1. Program wczytuje obrazy ze zbioru danych
2. Zmienia ich rozmiar do jednego standardu
3. Normalizuje wartości pikseli
4. Grupuje obrazy w batch’e
5. Trenuje sieć neuronową CNN
6. Zapisuje wytrenowany model do pliku
7. Umożliwia sprawdzenie własnego zdjęcia

Sieć patrzy na obraz warstwami:
- Warstwa 1 – wykrywa krawędzie 
- Warstwa 2 – wykrywa kształty (uszy, oczy) 
- Warstwa 3 – wykrywa fragmenty zwierzęcia (pysk, futro) 
- Warstwa 4 – decyduje czy to kot czy pies 

Na końcu zwracana jest liczba od 0 do 1: 
- blisko 0 -> kot (< 0.5 - kot)
- blisko 1 -> pies (> 0.5 - pies)


## Wymagania
Python 3.10+ \
Instalacja wymaganych bibliotek: \
`pip install tensorflow matplotlib numpy pillow`

## Zbiór danych
Aby pobrać ten sam zbiór danych co w projekcie należy skorzystać z bezpośredniego adresu URL w Pythonie (program pobiera go automatycznie z serwera Google Storage TensorFlow, a nie ręcznie z przeglądarki).
W celu pobrania tego samego zbioru danych należy: \
Uruchomić w folderze projektu skrypt pobierający dane (część programu lub jako osobny plik) \
Źródło danych (oficjalne repozytorium TensorFlow): `https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip` \
Po pobraniu w folderze projektu pojawi się katalog `cats_and_dogs_filtered/` jego ścieżkę należy przypisać do zmiennej base_dir w kodzie programu.

## Sprawdzenie własnego zdjęcia
W folderze projektu znjaduje się plik `pies.jpg` (lub dowolny inny obraz, pamiętać o zmianie nazwy) po uruchomieniu programu program wyświetli wynik klasyfikacji dla podanego zdjęcia.

## Tworzone pliki
Podczas działania projektu pojawią się:
- model.h5 - zapisany model (starszy format kompatybilności)
- model.keras - zapisany model (nowy format TensorFlow)

Są to wyuczone parametry sieci neuronowej. Dzięki nim nie trzeba trenować modelu ponownie.
Pliki są duże (~60 MB), dlatego nie są dodawane do repozytorium GitHub.
Każdy użytkownik powinien wygenerować je samodzielnie uruchamiając trening.
