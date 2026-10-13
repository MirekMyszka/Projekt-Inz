# Projekt-Inz

**Wykrywanie anomalii w szeregach czasowych z wykorzystaniem metod statystycznych i uczenia maszynowego** – projekt inżynierski.

Pełny opis projektu, stan prac i opis danych: [PROJEKT.md](PROJEKT.md). Szczegóły pobierania i czyszczenia danych: [docs/data_pipeline.md](docs/data_pipeline.md). Rodzaje anomalii w danych i plan doboru metod: [docs/plan_metod.md](docs/plan_metod.md). **Aktualny etap:** [docs/plan_etap_1.md](docs/plan_etap_1.md).

## Struktura

```
Projekt-Inz/
├── src/                    # kod (uruchamiany jako moduły: python -m src.<pakiet>.<skrypt>)
│   ├── config.py           # ścieżki i parametry zbiorów danych - jedno miejsce do zmian
│   ├── pobieranie/         # spy_alpaca.py, btcusdt_binance.py, nyc_taxi_nab.py
│   ├── czyszczenie/        # spy.py, btcusdt.py, nyc_taxi.py, wspolne.py (oznaczanie luk)
│   ├── eksploracja/        # podglad.py (statystyki + wykresy), przeglad_anomalii.py
│   ├── symulacje/          # [do zrobienia] szeregi syntetyczne + wstrzykiwanie anomalii
│   ├── metody/             # [do zrobienia] metody detekcji anomalii
│   └── ewaluacja/          # [do zrobienia] metryki i eksperymenty
├── dane/
│   ├── surowe/{spy,btcusdt,nyc_taxi}/   # dane prosto ze źródła
│   ├── clean/{spy,btcusdt,nyc_taxi}/    # dane po czyszczeniu + listy brakujących świec
│   └── cache/                  # pliki pośrednie (poza Gitem)
├── wyniki/wykresy/
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
```

## Praca z repo

```bash
git clone https://github.com/MirekMyszka/Projekt-Inz.git
cd Projekt-Inz
git checkout -b nazwa-branchu
```
