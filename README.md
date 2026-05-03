# Wykrywanie obrazów generowanych przez AI (Real vs AI)

Projekt koncentruje się na rozpoznawaniu, czy obraz jest:
- **prawdziwym zdjęciem** (`real`),
- **obrazem wygenerowanym przez AI** (`ai-generated`, np. GAN/diffusion),
- oraz (w osobnym scenariuszu) czy twarz na zdjęciu może być **deepfake**.

## Cel projektu

Zbudowanie i ocena modelu, który wykrywa obrazy syntetyczne oraz manipulacje twarzy, z naciskiem na:
- analizę artefaktów modeli generatywnych,
- odporność na nowe metody generowania,
- praktyczny kontekst bezpieczeństwa i dezinformacji.

## Zakres

1. **Klasyfikacja Real vs AI**  
   Model binarny klasyfikujący obrazy na `real` i `ai-generated`.

2. **Analiza artefaktów generatywnych**  
   Identyfikacja cech takich jak:
   - nienaturalne tekstury,
   - niespójne oświetlenie/cienie,
   - błędy w detalach (włosy, zęby, tło),
   - wzorce w domenie częstotliwości.

3. **Wykrywanie deepfake twarzy**  
   Pipeline: wykrycie twarzy -> klasyfikacja autentyczności twarzy.

4. **Ocena ryzyka i zagrożeń**  
   Analiza wpływu obrazów manipulowanych przez AI na:
   - dezinformację,
   - oszustwa tożsamościowe,
   - szantaż i szkody reputacyjne,
   - wiarygodność materiałów dowodowych.

## Proponowany pipeline

1. Przygotowanie i balansowanie danych.
2. Preprocessing (resize, normalizacja, augmentacje).
3. Trening modelu bazowego CNN/ViT.
4. Walidacja i test na nieznanych generatorach.
5. Interpretacja wyników (np. Grad-CAM, analiza błędów).

## Metryki

- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC
- (opcjonalnie) EER dla scenariuszy forensic

## Struktura repozytorium (plan)

```text
.
- data/                 # zbiory danych (lub skrypty ich pobierania)
- notebooks/            # eksperymenty i analiza
- src/                  # kod treningu, inferencji, ewaluacji
- reports/              # wyniki, wykresy, raport końcowy
- README.md
```

## Status

Repozytorium jest w fazie rozbudowy. Kolejne kroki:
- [ ] dodać źródła danych i ich opis,
- [ ] zaimplementować baseline model,
- [ ] uruchomić eksperymenty i porównać wyniki,
- [ ] przygotować raport o zagrożeniach i ograniczeniach.

## Etyka i ograniczenia

Systemy wykrywania obrazów AI mogą mieć ograniczoną skuteczność dla nowych generatorów i danych spoza rozkładu treningowego.  
Wyniki modelu należy traktować jako wsparcie analityczne, a nie niepodważalny dowód.

## Struktura danych

```text
data/real_vs_ai/
- train/
  - ai-generated/
  - real/
- val/
  - ai-generated/
  - real/
- test/
  - ai-generated/
  - real/
```

## Szybki start

1. Instalacja zależności:

```bash
pip install -r requirements.txt
```

Na Windows `requirements.txt` instaluje domyslnie build PyTorch z CUDA 12.8, zeby trening mogl korzystac z GPU NVIDIA zamiast wersji CPU-only.

2. Opcjonalnie: podziel dane na `train/val/test`:

```bash
python -m src.split_dataset --input-dir data/raw --output-dir data/real_vs_ai --train 0.7 --val 0.15 --test 0.15 --copy
```

Jesli dataset jest juz podzielony na `train/` i `test/`, mozna tylko wydzielic `val/` z `train/` bez kopiowania calego zbioru:

```bash
python -m src.split_dataset --input-dir data/real_vs_ai --layout pre_split --val-from-train 0.15
```

Ten wariant przenosi 15% plikow z kazdej klasy w `train/` do `val/` i zostawia `test/` bez zmian. Przy duzych zbiorach danych to bezpieczniejsza opcja niz duplikowanie plikow.

3. Ustaw parametry w `config.yaml`.

4. Trening baseline:

```bash
python -m src.train --config config.yaml
```

Przy treningu na GPU warto zostawic kilka workerow loadera, bo augmentacje obrazu odbywaja sie po stronie CPU. Domyslny config ustawia `num_workers: 6`, `persistent_workers: true`, `prefetch_factor: 3`, a takze `cudnn_benchmark: true` i `channels_last: true`, zeby karta nie czekala bezczynnie na kolejne batch'e przy stalym rozmiarze wejscia.

W trakcie treningu zapisywane sa checkpointy:
- `models/last_checkpoint.pt` po kazdej zakonczonej epoce
- `models/best_model.pt` dla najlepszego modelu
- `models/interrupted_checkpoint.pt` po przerwaniu treningu przez `Ctrl+C`

Domyslnie wlaczony jest tez `early stopping`, ktory zatrzyma trening, gdy metryka walidacyjna przestanie sie poprawiac przez kilka epok.

Wznowienie treningu:

```bash
python -m src.train --config config.yaml --resume models/last_checkpoint.pt
```

5. Predykcja dla jednego obrazu:

```bash
python -m src.predict --checkpoint models/best_model.pt --image path/to/image.jpg
```

6. Wstępna analiza twarzy (deepfake):

```bash
python -m src.deepfake_faces --checkpoint models/best_model.pt --image path/to/image.jpg
```

## Konfiguracja (`config.yaml`)

Najważniejsze pola:
- `data.data_dir`, `data.output_dir`
- `model.model_name`, `model.pretrained`, `model.image_size`
- `training.epochs`, `training.batch_size`, `training.learning_rate`, `training.num_workers`
- `runtime.torch_compile`, `runtime.compile_mode`, `runtime.compile_backend`

`src.train` ładuje parametry z `config.yaml`, a opcje CLI (np. `--epochs 20`) mogą je nadpisać.

## Grad-CAM

Przyklad zapisania mapy wyjasniajacej dla pojedynczego obrazu:

```bash
python -m src.predict --checkpoint models/best_model.pt --image path/to/image.jpg --save-cam reports/gradcam_example.jpg
```

## Test Odpornosci Na Jakosc

Po treningu mozna sprawdzic, jak model radzi sobie z kompresja JPEG, rozmyciem i downscale:

```bash
python -m src.robustness_eval --checkpoint models/best_model.pt
```

Raport zapisze sie domyslnie do `reports/robustness_eval.json`.

## Analiza Bledow

Po treningu mozna wyeksportowac pelny CSV z predykcjami oraz przykladowe false positive, false negative i poprawne klasyfikacje:

```bash
python -m src.error_analysis --checkpoint models/best_model.pt --split test --save-grad-cam
```

Mozna tez zawezic analize do wybranej klasy i posortowac przypadki wedlug trudnosci:

```bash
python -m src.error_analysis --checkpoint models/best_model.pt --split test --filter-class ai-generated --sort-mode hardest --save-grad-cam
```

Domyslnie raport zapisze sie do `reports/error_analysis/test/` i utworzy:
- `predictions.csv` z wszystkimi predykcjami dla wybranego splitu,
- `summary.json` z metrykami i macierza pomylek,
- `examples/` z wybranymi przypadkami do analizy i dokumentacji.

## Demo Webowe

Lokalne demo z uploadem obrazu, wynikiem klasyfikacji, mapa Grad-CAM oraz zakladka do analizy bledow:

```bash
python -m src.web_demo --checkpoint models/best_model.pt
```

Po uruchomieniu aplikacja bedzie dostepna lokalnie w przegladarce, domyslnie pod adresem `http://127.0.0.1:7860`.
W zakladce `Analiza bledow` mozna uruchomic eksport raportu dla `train`, `val` albo `test`, przefiltrowac przypadki po klasie i posortowac je np. trybem `hardest`, zeby od razu obejrzec najbardziej mylace false positive / false negative bez wychodzenia z demo.
