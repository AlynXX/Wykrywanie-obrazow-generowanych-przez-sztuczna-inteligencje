# Wykrywanie Obrazow Generowanych Przez AI

Projekt dotyczy wykrywania, czy obraz jest prawdziwym zdjeciem, czy zostal wygenerowany przez model AI. Repo zawiera juz domkniety baseline `real vs fake`, a dodatkowo jest przygotowane pod osobny etap twarzowy / deepfake oparty o cropy twarzy.

## Aktualny Stan

- Dziala trening klasyfikatora `real` / `fake`.
- Dziala wznowienie treningu z checkpointu.
- Dziala Grad-CAM dla pojedynczych obrazow.
- Dziala lokalne demo webowe w Gradio.
- Dziala test odpornosci na spadek jakosci obrazu.
- Dziala preprocessing datasetu twarzy z automatycznym cropowaniem.
- Jest gotowa konfiguracja treningu osobnego modelu twarzowego.
- Dziala benchmark porownujacy model globalny i model twarzowy na portretach.

## Najwazniejsze Wyniki

Najlepszy dotychczasowy baseline na zbiorze testowym:

- `accuracy=0.9339`
- `f1_macro=0.9339`
- `roc_auc=0.9831`

W nowszym treningu z mocniejszymi augmentacjami odpornosciowymi model uzyskal:

- `accuracy=0.9268`
- `f1_macro=0.9267`
- `roc_auc=0.9804`

Ten drugi wariant jest nieco slabszy na "czystym" tescie, ale lepiej nadaje sie do badan odpornosci na kompresje i degradacje obrazu.

## Podsumowanie Eksperymentow

Dwa najwazniejsze przebiegi treningu na etapie `real vs ai`:

| Wariant | Charakterystyka | Accuracy | F1 Macro | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Baseline | model ogolny, lzejsze augmentacje | 0.9339 | 0.9339 | 0.9831 |
| Robust | mocniejsze augmentacje jakosciowe | 0.9268 | 0.9267 | 0.9804 |

Wniosek praktyczny:

- baseline daje lepszy wynik na "czystym" tescie,
- wariant `robust` lepiej nadaje sie do analizy odpornosci na JPEG, blur i downscale,
- do pracy raportowej warto zachowac oba wyniki, bo razem dobrze pokazuja trade-off miedzy skutecznoscia i odpornoscia.

Syntetyczne podsumowanie etapu znajduje sie tez w `reports/PODSUMOWANIE_REAL_VS_AI.md`.

## Model

Obecny baseline korzysta z klasyfikatora obrazu zbudowanego na `timm`.

- architektura domyslna: `resnet18`
- wagi startowe: `pretrained=true`
- rozmiar wejscia: `224x224`
- klasy wyjsciowe: `fake`, `real`
- typ zadania: klasyfikacja calego obrazu, nie osobnej twarzy

To oznacza, ze model patrzy na caly kadr i uczy sie globalnych cech obrazu, a nie tylko obszaru twarzy. Z tego powodu dobrze nadaje sie jako pierwszy baseline `real vs ai`, ale nie powinien jeszcze byc traktowany jako docelowy detektor deepfake.

Najwazniejsze cechy treningu:

- loss: `CrossEntropyLoss`
- optimizer: `AdamW`
- metryka wyboru najlepszego checkpointu: `ROC-AUC` dla klasy `fake`
- wspierane wznowienie treningu z checkpointu
- wspierany `AMP` na GPU

Szczegoly ostatniego treningu sa zapisywane w `models/training_summary.json`.

## Ograniczenia Modelu

Model najlepiej radzi sobie ze zdjeciami fotograficznymi dobrej jakosci. Wyniki moga byc mniej wiarygodne dla:

- zdjec mocno skompresowanych przez komunikatory, np. Messenger,
- obrazow rozmytych lub o niskiej rozdzielczosci,
- portretow niskiej jakosci,
- ilustracji, anime i kadrów animowanych,
- danych spoza rozkladu treningowego.

Predykcje modelu nalezy traktowac jako wsparcie analityczne, a nie niepodwazalny dowod.

## Struktura Danych

Aktualny pipeline oczekuje danych w strukturze:

```text
data/real_vs_ai/
  train/
    fake/
    real/
  val/
    fake/
    real/
  test/
    fake/
    real/
```

Etap twarzowy korzysta z dwoch struktur danych:

```text
data/deepfake_faces/
  train/
    fake/
    real/
  val/
    fake/
    real/
  test/
    fake/
    real/
```

To sa oryginalne portrety lub klatki, z ktorych beda wycinane twarze.

Po preprocessingu powstaje osobny dataset cropow:

```text
data/deepfake_faces_crops/
  train/
    fake/
    real/
  val/
    fake/
    real/
  test/
    fake/
    real/
```

## Szybki Start

1. Instalacja zaleznosci:

```bash
pip install -r requirements.txt
```

Na Windows `requirements.txt` instaluje domyslnie build PyTorch z CUDA 12.8, zeby trening mogl korzystac z GPU NVIDIA zamiast wersji CPU-only.

2. Przygotowanie `val/`, jesli dataset ma tylko `train/` i `test/`:

```bash
python -m src.split_dataset --input-dir data/real_vs_ai --layout pre_split --val-from-train 0.15
```

3. Trening:

```bash
python -m src.train --config config.yaml
```

4. Wznowienie treningu:

```bash
python -m src.train --config config.yaml --resume models/last_checkpoint.pt
```

5. Predykcja dla jednego obrazu:

```bash
python -m src.predict --checkpoint models/best_model.pt --image path/to/image.jpg
```

6. Zapisanie mapy Grad-CAM:

```bash
python -m src.predict --checkpoint models/best_model.pt --image path/to/image.jpg --save-cam reports/gradcam_example.jpg
```

7. Test odpornosci na pogorszenie jakosci obrazu:

```bash
python -m src.robustness_eval --checkpoint models/best_model.pt
```

Raport zapisze sie domyslnie do `reports/robustness_eval.json`.

## Etap Twarzowy

1. Przygotowanie cropow twarzy z surowego datasetu portretow:

```bash
python -m src.deepfake_faces prepare-dataset --input-dir data/deepfake_faces --output-dir data/deepfake_faces_crops
```

Domyslnie skrypt zachowuje tylko najwieksza twarz na obrazie, dodaje margines wokol bboxu i zapisuje manifest do `data/deepfake_faces_crops/face_dataset_manifest.json`.

2. Analiza pojedynczego obrazu z wykryciem twarzy:

```bash
python -m src.deepfake_faces --checkpoint models/best_model.pt --image path/to/portrait.jpg --save-annotated-image reports/faces_preview.jpg
```

Komenda dziala tez w trybie jawnym:

```bash
python -m src.deepfake_faces inspect --checkpoint models/best_model.pt --image path/to/portrait.jpg
```

3. Trening osobnego modelu twarzowego:

```bash
python -m src.train --config config_faces.yaml
```

Checkpointy i podsumowanie treningu zapisza sie domyslnie do `models/faces/`.

4. Porownanie modelu globalnego i twarzowego na portretach:

```bash
python -m src.compare_face_models --global-checkpoint models/best_model.pt --face-checkpoint models/faces/best_model.pt --data-dir data/deepfake_faces --split test
```

Raport porownawczy zapisze sie domyslnie do `reports/face_model_comparison.json`, a pelny CSV obok niego.

5. Przygotowanie malego datasetu adaptacyjnego `hard fakes + old real`:

```bash
python -m src.prepare_adaptation_dataset --fake-dir data/new_dataset/hard_fakes --real-dir data/deepfake_faces_crops/train/real --output-dir data/hard_fakes_adaptation --copy
```

Domyslnie skrypt przygotuje:
- `train/fake=500`, `val/fake=100`, `test/fake=100`
- `train/real=1000`, `val/real=200`, `test/real=200`

Wszystkie przypisania zapisze tez do `adaptation_manifest.json`.

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

Jesli dostepny jest tez model twarzowy, demo moze pokazac oba etapy projektu naraz:

```bash
python -m src.web_demo --checkpoint models/best_model.pt --face-checkpoint models/faces/best_model.pt
```

Po uruchomieniu aplikacja bedzie dostepna lokalnie w przegladarce, domyslnie pod adresem `http://127.0.0.1:7860`.
W zakladce `Pojedynczy obraz` zobaczysz wtedy wynik modelu globalnego dla calego kadru oraz osobny wynik modelu twarzowego dla najwiekszej wykrytej twarzy.
W zakladce `Analiza bledow` mozna uruchomic eksport raportu dla `train`, `val` albo `test`, przefiltrowac przypadki po klasie i posortowac je np. trybem `hardest`, zeby od razu obejrzec najbardziej mylace false positive / false negative bez wychodzenia z demo.
