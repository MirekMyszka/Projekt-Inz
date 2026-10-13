"""Pobieranie szeregu NYC Taxi z Numenta Anomaly Benchmark (NAB) razem z etykietami anomalii.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.pobieranie.nyc_taxi_nab

Źródło: https://github.com/numenta/NAB (licencja MIT), wersja przypięta do commita z src/config.py.
- dane:     data/realKnownCause/nyc_taxi.csv  - liczba pasażerów taksówek w NYC w przedziałach 30 min
            (2014-07-01 - 2015-01-31), zagregowana przez Numenta z danych NYC Taxi and Limousine Commission,
- etykiety: labels/combined_windows.json (okna anomalii) i labels/combined_labels.json (punkty anomalii).
Bez klucza API. Wynik:
    dane/surowe/nyc_taxi/nyc_taxi_nab_pobrano_<data>.csv
    dane/surowe/nyc_taxi/nyc_taxi_etykiety_nab_pobrano_<data>.json
"""
import json
from datetime import date

import requests

from src.config import NYC_TAXI

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{sciezka}"


def pobierz(sciezka: str) -> requests.Response:
    url = RAW_URL.format(repo=NYC_TAXI["repo"], commit=NYC_TAXI["commit"], sciezka=sciezka)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r


if __name__ == "__main__":
    out_dir = NYC_TAXI["surowe_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dzis = date.today()
    klucz = NYC_TAXI["plik_nab"]

    csv_path = out_dir / f"nyc_taxi_nab_pobrano_{dzis}.csv"
    csv_path.write_bytes(pobierz(f"data/{klucz}").content)

    etykiety = {
        "zrodlo": f"https://github.com/{NYC_TAXI['repo']}",
        "commit": NYC_TAXI["commit"],
        "plik": klucz,
        "okna": pobierz("labels/combined_windows.json").json()[klucz],
        "punkty": pobierz("labels/combined_labels.json").json()[klucz],
    }
    json_path = out_dir / f"nyc_taxi_etykiety_nab_pobrano_{dzis}.json"
    json_path.write_text(json.dumps(etykiety, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Zapisano dane do: {csv_path}")
    print(f"Zapisano etykiety ({len(etykiety['okna'])} okien, {len(etykiety['punkty'])} punktów) do: {json_path}")
