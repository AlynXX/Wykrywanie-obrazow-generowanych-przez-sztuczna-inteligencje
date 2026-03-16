# Plan Implementacji Systemu (Budżet czasowy: ~30h)

Projekt koncentruje się na budowie systemu do wykrywania obrazów generowanych przez AI oraz manipulacji typu deepfake, w kontekście zabezpieczania aplikacji, weryfikacji tożsamości i wykrywania zagrożeń (np. na przedmiocie "Analiza obrazów z elementami sztucznej inteligencji w detekcji zagrożeń"). Aby zmieścić się w zakładanym czasie 30h, podzieliliśmy proces na zadania z priorytetami `(MVP / Core)` na początek oraz `(Opcjonalne / Ext)` jako usprawnienia, jeśli starczy czasu.

---

## Faza 1: Obsługa zestawów danych i preaktywna obrona `[MVP]`
*Zakładany czas: ~5-6h*
Zbiory obrazów dla modeli różniących się technologicznie (GAN vs. Modele dyfuzyjne).
1. `dataset.py`: wdrożenie obsługi struktury danych `Real` / `AI`.
2. Opracowanie silnego bloku **Augmentacji zacierających**. W prawdziwych zagrożeniach sprawcy wprowadzają modyfikacje minimalizujące widoczność artefaktów. Należy dodać kompresję stratną JPG, szum Gaussa i rozmycie przed podaniem danych na wejście modelu uczącego. Mechanizm uodporni na modyfikacje.

## Faza 2: Pierwszy detektor: Analiza Pełnych Obrazów (Real vs. AI) `[MVP]`
*Zakładany czas: ~6-8h*
Klasyfikator globalnego zdjęcia wycelowany do rozpoznawania zdjęć wyciągniętych ze środowisk takich jak Midjourney/Stable Diffusion (często fałszywe sceny bitew, nagłe sytuacje społeczne itp.).
1. Użycie pakietu `timm` (lub bezpośrednio torchvision). Trening klasycznego ResNet/EfficientNet na całości.
2. Logika `train.py`: pętla ucząca, logowanie metryk i zapis wag dla najlepszego modelu. Do tego ewaluacja `Accuracy / ROC-AUC`.

## Faza 3: Drugi Detektor: Wykrywacz zagrożeń Deepfake `[MVP]`
*Zakładany czas: ~7-9h*
Obrona przed zagrożeniami kradzieży tożsamości.
1. `deepfake_faces.py`: wdrożenie detekcji twarzy (Face Cropping) na wejściu. Może posłużyć model `MTCNN`, `RetinaFace` lub po prostu najszybszy blok Haar Cascades dostępny z `OpenCV`.
2. Przycięte obrazy (współrzędne tzw. Region Of Interest - ROI) podawane są z powrotem do wyspecjalizowanego klasyfikatora, wyczulonego wyłącznie na błędy renderowania oczu/nosów/włosów. Pomoże to ustrzec algorytm przed zwracaniem uwagi na tło, a skupi na manipulowanym człowieku.

## Faza 4: Explainable AI i Prezentacja (Podejmowanie Decyzji) `[MVP]`
*Zakładany czas: ~5h*
Aspekt bardzo mile widziany na zajęciach: Nie wystarczy pokazać, że "to jest fałszywka". Należy wiedzieć "dlaczego".
1. `predict.py` zostanie rozszerzone o rysunek powstrzymanej fali manipulacji – użycie mapy aktywacji **Grad-CAM**. Stworzenie kolorowej nałożonej "mapy termicznej" odsłaniającej, na podstawie których obszarów na zdjęciu model powziął swoją dedukcje (wyłapane błędy palców rąk, dziwny kontur żuchwy u deepfake'u itp.).
2. Podsumowanie statystyczne eksperymentów dla dokumentacji.

---

## Etap 5: "Furtki" (Zadania Dodatkowe na sam koniec) `[Opcjonalne / Ext]`
*Zakładany czas: Pozostałości budżetowe*
* Zastosowanie detekcji z przestrzeni częstotliwości (np. zjawiska szumu widmowego wygenerowanego przez AI).
* *Ensembling*: zapięcie dwóch różnych architektur (np. ViT i CNN) w głosowanie większościowe, jeśli skuteczność pojedynczego detektora na nowych danych byłaby zbyt niska.
* Logowanie eksperymentów w popularnych narzędziach ułatwiających późniejszą obronę decyzji analitycznych (np. MLflow / TensorBoard).
