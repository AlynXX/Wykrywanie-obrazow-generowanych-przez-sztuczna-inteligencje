# Wykrywanie Obrazow Generowanych Przez AI

Projekt dotyczy wykrywania, czy obraz jest prawdziwym zdjeciem, czy zostal wygenerowany przez model AI. Obecna wersja repo skupia sie przede wszystkim na baseline `real vs fake`, a etap deepfake twarzy jest przygotowany jako nastepny krok rozwoju.

## Aktualny Stan

- Dziala trening klasyfikatora `real` / `fake`.
- Dziala wznowienie treningu z checkpointu.
- Dziala Grad-CAM dla pojedynczych obrazow.
- Dziala lokalne demo webowe w Gradio.
- Dziala test odpornosci na spadek jakosci obrazu.
- Etap deepfake twarzy nie jest jeszcze domkniety jako osobny, finalny pipeline.

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

8. Uruchomienie lokalnego demo:

```bash
python -m src.web_demo --checkpoint models/best_model.pt
```

Po uruchomieniu aplikacja jest dostepna lokalnie pod adresem `http://127.0.0.1:7860`.

## Trening I Runtime

Repo ma juz przygotowane mechanizmy, ktore ulatwiaja dluzsze eksperymenty:

- automatyczny dobor `batch_size`,
- `AMP` na GPU,
- checkpointy `best_model.pt`, `last_checkpoint.pt`, `interrupted_checkpoint.pt`,
- `early stopping`,
- zapis podsumowania do `models/training_summary.json`.

Przy treningu na GPU wazne sa tez ustawienia loadera. Domyslny config ustawia:

- `num_workers: 6`
- `persistent_workers: true`
- `prefetch_factor: 3`
- `cudnn_benchmark: true`
- `channels_last: true`

To pomaga lepiej wykorzystac GPU przy stalym rozmiarze wejscia `224x224`.

## Interpretacja Wynikow

Demo i CLI zwracaja:

- etykiete `real` lub `fake`,
- pewnosc predykcji,
- mape Grad-CAM,
- ostrzezenie o niskiej jakosci obrazu, jesli obraz jest rozmyty, zbyt maly albo ma bardzo niski kontrast.

Warning o jakosci nie oznacza, ze obraz jest falszywy. Oznacza tylko, ze wynik modelu moze byc mniej stabilny.

## Test Odpornosci

`src.robustness_eval` sprawdza model na kilku profilach degradacji:

- `clean`
- `jpeg_low`
- `jpeg_extreme`
- `blur_light`
- `blur_strong`
- `downscale_light`
- `downscale_strong`
- `mixed_quality`

Raport zapisuje sie domyslnie do `reports/robustness_eval.json`.

Najwazniejsze obserwacje z aktualnego raportu:

- degradacje lekkie sa znoszone dobrze,
- najwiekszy spadek daje `jpeg_extreme`,
- kombinacja kilku degradacji (`mixed_quality`) nadal obniza skutecznosc wyraznie bardziej niz pojedyncze lekkie zaklocenie.

## Struktura Repozytorium

```text
.
  config.yaml
  data/
  models/
  reports/
  src/
    auto_batch.py
    dataset.py
    deepfake_faces.py
    inference.py
    model.py
    predict.py
    robustness_eval.py
    split_dataset.py
    train.py
    web_demo.py
  PLAN_PROJEKTU.md
  README.md
  requirements.txt
```

## Co Dalej

Najblizszy plan rozwoju projektu:

- przygotowanie osobnego baseline'u dla twarzy i deepfake,
- testy na zewnetrznych zestawach twarzy,
- dalsza analiza bledow i przypadkow spoza rozkladu treningowego.
