# Projekt-Inz

**Wykrywanie anomalii w szeregach czasowych z wykorzystaniem metod statystycznych i uczenia maszynowego** – projekt inżynierski.

Pełny opis projektu, stan prac i opis danych: [PROJEKT.md](PROJEKT.md). Szczegóły pobierania i czyszczenia danych: [docs/data_pipeline.md](docs/data_pipeline.md). Rodzaje anomalii w danych i plan doboru metod: [docs/plan_metod.md](docs/plan_metod.md). **Aktualny etap:** [docs/plan_etap_1.md](docs/plan_etap_1.md). **Wyniki:** [S1 – biały szum](docs/wyniki_s1.md), [S2 – AR(1)](docs/wyniki_s2.md), [S3 – szum + sinus](docs/wyniki_s3.md), [porównanie S1–S3](docs/porownanie_s1_s3.md), [dane rzeczywiste – etap 2](docs/wyniki_rzeczywiste.md); na start: [kompendium wiedzy](docs/kompendium.md). **Jak działa każda metoda:** [docs/metody.md](docs/metody.md).

## Struktura

```
Projekt-Inz/
├── src/                    # kod (uruchamiany jako moduły: python -m src.<pakiet>.<skrypt>)
│   ├── config.py           # ścieżki i parametry zbiorów danych - jedno miejsce do zmian
│   ├── pobieranie/         # spy_alpaca.py, btcusdt_binance.py, nyc_taxi_nab.py
│   ├── czyszczenie/        # spy.py, btcusdt.py, nyc_taxi.py, wspolne.py (oznaczanie luk)
│   ├── eksploracja/        # podglad.py (statystyki + wykresy), przeglad_anomalii.py, raport_danych.py (kontrola jakości)
│   ├── symulacje/          # generator.py, anomalie.py (S1, S2, S3), pilot_etap1.py
│   ├── metody/             # baza.py (wspólny interfejs), statystyczne.py, fragmenty.py, ml.py
│   ├── ewaluacja/          # metryki.py, eksperyment.py, eksperyment_s1/s2/s3.py, wykresy_s1/s2/s3.py, walidacja_progow.py, warianty_metod.py, porownanie.py
│   └── rzeczywiste/        # etap 2: nyc_taxi.py, rynki.py, studia.py, katalog.py, wspolne.py, wykresy_rzeczywiste.py
├── dane/
│   ├── surowe/{spy,btcusdt,nyc_taxi}/   # dane prosto ze źródła
│   ├── clean/{spy,btcusdt,nyc_taxi}/    # dane po czyszczeniu + listy brakujących świec
│   └── cache/                  # pliki pośrednie (poza Gitem)
├── wyniki/
│   ├── s1/, s2/, s3/       # wyniki eksperymentów (CSV) + podsumowanie.md
│   ├── porownanie/         # porównanie S1–S3
│   ├── rzeczywiste/        # etap 2: dane rzeczywiste
│   ├── walidacja_progow.*  # sprawdzenie progów na 1 000 nowych szeregach
│   └── wykresy/
├── tests/                  # testy (pytest)
└── docs/
```

## Konfiguracja środowiska

```bash
python -m venv ..\.venv
..\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # i wpisać klucze Alpaca (potrzebne tylko do pobierania SPY)
```

Plik `.env` jest ignorowany przez Git – prawdziwe klucze nie trafiają na GitHub.

## Uruchomienie (z katalogu `Projekt-Inz`)

```bash
# SPY (Alpaca, wymaga kluczy w .env)
python -m src.pobieranie.spy_alpaca
python -m src.czyszczenie.spy

# BTCUSDT (Binance, bez klucza)
python -m src.pobieranie.btcusdt_binance
python -m src.czyszczenie.btcusdt

# NYC Taxi z etykietami anomalii (Numenta Anomaly Benchmark, bez klucza)
python -m src.pobieranie.nyc_taxi_nab
python -m src.czyszczenie.nyc_taxi

# podgląd i wykresy wszystkich zbiorów
python -m src.eksploracja.podglad
python -m src.eksploracja.raport_danych   # niezależna kontrola jakości i właściwości danych (wyniki/dane/)

# scenariusz S1 (biały szum): eksperyment symulacyjny (ok. 1 h na 2 rdzeniach) i wykresy
python -m src.ewaluacja.eksperyment_s1 --szybko   # najpierw szybkie sprawdzenie (ok. 6 min)
python -m src.ewaluacja.eksperyment_s1
python -m src.ewaluacja.wykresy_s1

# scenariusz S2 (AR(1)): eksperyment (ok. 35 min) i wykresy
python -m src.ewaluacja.eksperyment_s2 --szybko   # szybkie sprawdzenie (ok. 4 min)
python -m src.ewaluacja.eksperyment_s2
python -m src.ewaluacja.wykresy_s2

# scenariusz S3 (szum + sinus): eksperyment (ok. 4 h; można po części: glowny, okno, tlo, dodatkowa) i wykresy
python -m src.ewaluacja.eksperyment_s3 --szybko   # szybkie sprawdzenie (ok. 25 min)
python -m src.ewaluacja.eksperyment_s3
python -m src.ewaluacja.wykresy_s3

# sprawdzenie progów alarmu S1–S3 na 1 000 nowych szeregach bez anomalii (ok. 1,5 h)
python -m src.ewaluacja.walidacja_progow

# porównanie S1–S3: przeniesienie metod między scenariuszami (ok. 2 h; można po komórkach, np. "S3>S1") i wykresy
python -m src.ewaluacja.porownanie
python -m src.ewaluacja.wykresy_porownanie

# etap 2 – dane rzeczywiste: NYC Taxi, SPY i BTC (szeregi dzienne), studia przypadków 5-min, wykresy
python -m src.rzeczywiste.nyc_taxi
python -m src.rzeczywiste.rynki
python -m src.rzeczywiste.studia
python -m src.rzeczywiste.wykresy_rzeczywiste

# testy
python -m pytest tests -q
```

## Praca z repo

```bash
git clone https://github.com/MirekMyszka/Projekt-Inz.git
cd Projekt-Inz
git checkout -b nazwa-branchu
```
