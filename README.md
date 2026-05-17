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

Nowy, mocniejszy eksperyment `ConvNeXt v2` moze korzystac z curated miksu wielozrodlowego:

```text
data/deepfake_faces_v2/
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
Mozesz tez przygotowac szersze cropy portretowe:

```bash
python -m src.deepfake_faces prepare-dataset --input-dir data/deepfake_faces --output-dir data/deepfake_portraits --crop-style portrait
```

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

Aktualna konfiguracja twarzowa korzysta z `convnext_tiny`.
Checkpointy i podsumowanie treningu zapisza sie domyslnie do `models/faces_convnext/`.

3a. Zlozenie curated datasetu `ConvNeXt v2` z duzego, roznorodnego zbioru:

Najpierw umiesc wybrane zrodla pod katalogiem `data/multisource_raw/`, zgodnie z plikiem
`curated_faces_v2_sources.yaml`, a potem uruchom:

```bash
python -m src.prepare_curated_face_dataset --spec curated_faces_v2_sources.yaml --copy
```

Domyslnie skrypt sklada kontrolowany miks z:
- `FFHQ`
- `Multiface`
- `140k-real-vs-fake`
- `Deep-vs-Real`
- `deepfake-vs-real-60k`
- `stable_diffusion`
- `dalle-generated`
- `real-vs-hardfakes`

Wynik trafi do `data/deepfake_faces_v2/`, a pelny manifest do
`data/deepfake_faces_v2/curated_dataset_manifest.json`.

3b. Trening `ConvNeXt v2` na curated miksie:

```bash
python -m src.train --config config_faces_v2.yaml
```

Checkpointy zapisza sie do `models/faces_convnext_v2/`.

4. Porownanie modelu globalnego i twarzowego na portretach:

```bash
python -m src.compare_face_models --global-checkpoint models/best_model.pt --face-checkpoint models/faces_convnext/best_model.pt --data-dir data/deepfake_faces --split test
```

Raport porownawczy zapisze sie domyslnie do `reports/face_model_comparison.json`, a pelny CSV obok niego.
Do eksperymentu z szerszym cropem portretowym mozna uzyc:

```bash
python -m src.compare_face_models --global-checkpoint models/best_model.pt --face-checkpoint models/faces_convnext/best_model.pt --data-dir data/deepfake_faces --split test --crop-style portrait
```

5. Przygotowanie malego datasetu adaptacyjnego `hard fakes + old real`:

```bash
python -m src.prepare_adaptation_dataset --fake-dir data/new_dataset/hard_fakes --real-dir data/deepfake_faces_crops/train/real --output-dir data/hard_fakes_adaptation --copy
```

Domyslnie skrypt przygotuje:
- `train/fake=500`, `val/fake=100`, `test/fake=100`
- `train/real=1000`, `val/real=200`, `test/real=200`

Wszystkie przypisania zapisze tez do `adaptation_manifest.json`.

6. Fine-tuning modelu twarzowego na `hard fakes`:

```bash
python -m src.train --config config_hard_fakes.yaml --init-checkpoint models/faces_convnext/best_model.pt
```

Ta komenda startuje od wag najlepszego modelu twarzowego, ale nie przenosi historii, optimizera ani starego `best_val_score`, wiec nadaje sie do czystego eksperymentu adaptacyjnego.

7. Strojenie progu decyzyjnego dla modelu adaptacyjnego:

```bash
python -m src.tune_threshold --checkpoint models/faces_convnext_hard_adapt/best_model.pt --config config_hard_fakes.yaml --tune-split val --eval-split test --metric f1_positive
```

Skrypt zapisze raport do `reports/threshold_tuning/` i poda najlepszy prog dla klasy `fake`.

8. Uzycie progu w predykcji lub GUI:

```bash
python -m src.predict --checkpoint models/faces_convnext_hard_adapt/best_model.pt --image path/to/image.jpg --threshold 0.62
```

```bash
python -m src.web_demo --checkpoint models/best_model.pt --face-checkpoint models/faces_convnext_hard_adapt/best_model.pt --face-threshold 0.62
```

9. Przygotowanie zewnetrznego benchmarku Gemini:

```bash
python -m src.prepare_gemini_benchmark --fake-dir data/gemini_benchmark_sources/fake --real-dir data/gemini_benchmark_sources/real --output-dir data/gemini_benchmark --copy
```

Domyslnie skrypt zbuduje split `test/` i dobierze tyle samo `real`, ile dostepnych `fake`.

10. Ewaluacja na benchmarku Gemini:

```bash
python -m src.error_analysis --checkpoint models/best_model.pt --config config_gemini_benchmark.yaml --split test --output-dir reports/gemini_global
```

```bash
python -m src.compare_face_models --global-checkpoint models/best_model.pt --face-checkpoint models/faces_convnext_hard_adapt/best_model.pt --data-dir data/gemini_benchmark --split test --crop-style portrait --output-path reports/gemini_face_vs_global.json
```

## Pilotowy Benchmark OOD

Jesli chcesz szybko sprawdzic, czy model gubi najnowsze generatory bez recznego budowania duzego datasetu, w repo jest przygotowany maly workflow pilota:

1. Gotowa pula promptow startowych:

```text
pilot_ood_prompt_pool.csv
```

2. Prosty arkusz do odhaczania probek:

```text
pilot_ood_samples_template.csv
```

3. Domyslna struktura surowych folderow:

```text
data/ood_sources/
  gpt_fake/
  nano2_fake/
  real_matched/
```

4. Domyslna specyfikacja pilota:

```text
pilot_ood_benchmark_sources.yaml
```

Domyslnie pilot sklada benchmark `test` z:
- `gpt_fake=20`
- `nano2_fake=20`
- `real_matched=40`

5. Zlozenie benchmarku jedna komenda:

```bash
python -m src.prepare_ood_benchmark --spec pilot_ood_benchmark_sources.yaml --copy
```

Wynik trafi do:

```text
data/ood_pilot_benchmark/
  test/
    fake/
    real/
```

Skrypt zapisze tez manifest do:

```text
data/ood_pilot_benchmark/ood_benchmark_manifest.json
```

Nazwy eksportowanych plikow zachowuja prefiks zrodla, np. `gpt__...` albo `nano2__...`, wiec po ewaluacji latwo policzyc wyniki per generator.

6. Ewaluacja modelu globalnego na pilocie:

```bash
python -m src.error_analysis --checkpoint models/best_model.pt --config config_gemini_benchmark.yaml --data-dir data/ood_pilot_benchmark --split test --output-dir reports/ood_pilot_global
```

7. Porownanie modelu globalnego i twarzowego:

```bash
python -m src.compare_face_models --global-checkpoint models/best_model.pt --face-checkpoint models/faces_convnext_hard_adapt/best_model.pt --data-dir data/ood_pilot_benchmark --split test --crop-style portrait --output-path reports/ood_pilot_face_vs_global.json
```

## Opcja A: Adaptacja Modelu Globalnego

Jesli chcesz szybko pokazac postep na nowych generatorach, najprostsza sciezka to dostrojenie modelu globalnego na nowych `GPT/Nano2 + real`, ale bez naruszania zamrozonego benchmarku `ood_pilot_benchmark`.

1. Zostaw `data/ood_pilot_benchmark/` tylko do finalnej ewaluacji.

2. Zbierz osobny surowy material adaptacyjny do:

```text
data/global_adaptation_sources/
  gpt_fake/
  nano2_fake/
  real_matched/
```

To powinny byc nowe obrazy, inne niz te uzyte w `ood_pilot_benchmark`.

3. Specyfikacja adaptacji jest gotowa w:

```text
global_adaptation_sources.yaml
```

Domyslnie laczy:
- nowe `gpt_fake`
- nowe `nano2_fake`
- nowe `real_matched`
- ograniczona domieszke starego `legacy_fake` z `data/real_vs_ai/train/fake`
- ograniczona domieszke starego `legacy_real` z `data/real_vs_ai/train/real`

4. Zlozenie datasetu adaptacyjnego:

```bash
python -m src.prepare_global_adaptation_dataset --spec global_adaptation_sources.yaml --copy
```

Wynik trafi do:

```text
data/global_adaptation/
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

5. Gotowy config fine-tuningu:

```text
config_global_adaptation.yaml
```

6. Fine-tuning modelu globalnego od istniejacego checkpointu:

```bash
python -m src.train --config config_global_adaptation.yaml --init-checkpoint models/best_model.pt
```

Checkpoint adaptacyjny zapisze sie domyslnie do:

```text
models/global_ood_adapt/
```

7. Opcjonalne strojenie progu dla nowego modelu:

```bash
python -m src.tune_threshold --checkpoint models/global_ood_adapt/best_model.pt --config config_global_adaptation.yaml --tune-split val --eval-split test --metric f1_positive
```

8. Finalna ewaluacja na zamrozonym benchmarku pilota:

```bash
python -m src.error_analysis --checkpoint models/global_ood_adapt/best_model.pt --config config_gemini_benchmark.yaml --data-dir data/ood_pilot_benchmark --split test --output-dir reports/ood_pilot_global_adapted
```

Jesli po strojeniu progu chcesz uzyc konkretnej wartosci dla klasy `fake`, mozesz potem podac ja jawnie w `src.predict` albo `src.web_demo` przez `--threshold`.

## Kolejny Krok: Adaptacja Modelu Twarzowego

Jesli po etapie diagnostycznym uznasz, ze glownym kierunkiem rozwoju powinien byc model twarzowy, repo ma teraz tez gotowy workflow pod ten wariant.

1. Zbierz surowe portrety do:

```text
data/face_adaptation_sources_raw/
  fake/
  real/
```

To powinny byc nowe obrazy `GPT/Nano2 + real`, inne niz benchmark testowy.

2. Wytnij twarze do osobnego katalogu cropow:

```bash
python -m src.deepfake_faces prepare-dataset --input-dir data/face_adaptation_sources_raw --output-dir data/face_adaptation_sources_crops --crop-style face
```

Jesli chcesz porownac szersze ujecia, mozesz zamiast tego uzyc `--crop-style portrait`, ale najczystszy pierwszy eksperyment adaptacyjny warto zrobic na `face`.

3. Specyfikacja miksu adaptacyjnego:

```text
face_adaptation_sources.yaml
```

Domyslnie laczy:
- nowe cropy `fake` z `data/face_adaptation_sources_crops/fake`
- nowe cropy `real` z `data/face_adaptation_sources_crops/real`
- domieszke starszych danych z `data/deepfake_faces_v2/train`

4. Zlozenie datasetu adaptacyjnego:

```bash
python -m src.prepare_face_adaptation_dataset --spec face_adaptation_sources.yaml --copy
```

Wynik trafi do:

```text
data/face_adaptation/
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

5. Gotowy config fine-tuningu modelu twarzowego:

```text
config_face_adaptation.yaml
```

6. Fine-tuning od obecnego checkpointu `ConvNeXt v2`:

```bash
python -m src.train --config config_face_adaptation.yaml --init-checkpoint models/faces_convnext_v2/best_model.pt
```

Checkpoint zapisze sie domyslnie do:

```text
models/faces_convnext_v2_adapt/
```

7. Finalna ewaluacja na benchmarku portretowym:

```bash
python -m src.compare_face_models --global-checkpoint models/best_model.pt --face-checkpoint models/faces_convnext_v2_adapt/best_model.pt --data-dir data/ood_pilot_benchmark --split test --crop-style face --output-path reports/ood_pilot_face_vs_global_v2_adapt.json
```

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
python -m src.web_demo --checkpoint models/best_model.pt --face-checkpoint models/faces_convnext/best_model.pt
```

Mozesz tez przetestowac szerszy crop portretowy:

```bash
python -m src.web_demo --checkpoint models/best_model.pt --face-checkpoint models/faces_convnext_hard_adapt/best_model.pt --face-threshold 0.62 --face-crop-style portrait
```

Po uruchomieniu aplikacja bedzie dostepna lokalnie w przegladarce, domyslnie pod adresem `http://127.0.0.1:7860`.
W zakladce `Pojedynczy obraz` zobaczysz wtedy wynik modelu globalnego dla calego kadru oraz osobny wynik modelu twarzowego dla najwiekszej wykrytej twarzy.
W zakladce `Analiza bledow` mozna uruchomic eksport raportu dla `train`, `val` albo `test`, przefiltrowac przypadki po klasie i posortowac je np. trybem `hardest`, zeby od razu obejrzec najbardziej mylace false positive / false negative bez wychodzenia z demo.
