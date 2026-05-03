# Plan Projektu

## Cel

Zbudowac system, ktory potrafi odroznic prawdziwe zdjecia od obrazow generowanych przez AI, a w kolejnym etapie rozszerzyc go o analize twarzy i przypadkow typu deepfake.

## Etap 1: Real Vs AI

Status: `w duzej mierze zakonczony`

Zrealizowane elementy:

- przygotowanie loadera danych `train / val / test`,
- skrypt do wydzielania `val` ze struktury `train / test`,
- trening klasyfikatora obrazu z wykorzystaniem `timm`,
- zapis checkpointow i wznowienie treningu,
- `AMP`, auto batch i optymalizacje pod GPU,
- `early stopping`,
- Grad-CAM dla pojedynczej predykcji,
- lokalne demo webowe,
- benchmark odpornosci na spadek jakosci obrazu.

Najwazniejsze obserwacje:

- model dobrze radzi sobie z normalnymi zdjeciami dobrej jakosci,
- pogorszenie jakosci obrazu obniza skutecznosc,
- zdjecia z komunikatorow i portrety niskiej jakosci moga byc mylone z `fake`,
- anime i ilustracje sa poza glownym zakresem obecnego modelu.

## Etap 2: Analiza Ograniczen

Status: `czesciowo wykonany`

Zrealizowane elementy:

- warning o niskiej jakosci obrazu w demo,
- test `robustness_eval` dla JPEG, blur i downscale,
- reczna analiza blednych predykcji na zdjeciach spoza datasetu.

Do dopracowania:

- audyt bardzo duzych lub problematycznych obrazow w datasetcie,
- bardziej formalny zestaw testow zewnetrznych,
- ewentualny stan `niepewne` w GUI.

## Etap 3: Deepfake Twarzy

Status: `nastepny glowny krok`

Plan:

1. Przygotowac lub uporzadkowac dataset twarzy.
2. Dopracowac `src/deepfake_faces.py` pod wykrywanie i cropowanie twarzy.
3. Wytrenowac osobny baseline dla twarzy `real / fake`.
4. Sprawdzic, czy model twarzowy rzeczywiscie poprawia wyniki dla portretow.

Powod:

Obecny model ogolny bywa zawodny na portretach niskiej jakosci, wiec etap twarzowy powinien byc osobnym, wyspecjalizowanym modulem zamiast tylko "testem na twarzach" obecnego klasyfikatora.

## Etap 4: Dokumentacja Koncowa

Status: `jeszcze nie finalizowac`

Na teraz warto utrzymywac:

- `README.md` jako dokumentacje techniczna repo,
- `reports/PODSUMOWANIE_REAL_VS_AI.md` jako robocze podsumowanie etapu wynikowego,
- ten plan jako aktualny stan projektu.

Pelna dokumentacja w LaTeX powinna powstac dopiero wtedy, gdy:

- etap `real vs ai` bedzie uznany za zamkniety,
- baseline twarzowy bedzie juz gotowy albo swiadomie odlozony,
- bedzie jasne, ktore wyniki trafia do finalnego raportu.

## Biezace Priorytety

1. Rozpoczac etap twarzowy / deepfake.
2. Zebrac pierwsze wyniki na cropach twarzy.
3. Porownac zachowanie modelu ogolnego i modelu twarzowego.
4. Dopiero potem wracac do pelnej dokumentacji raportowej.
