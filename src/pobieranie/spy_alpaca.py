"""Pobieranie świec 5-min SPY z Alpaca Market Data API.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.pobieranie.spy_alpaca

Wymaga kluczy APCA_API_KEY_ID i APCA_API_SECRET_KEY w pliku .env (wzór: .env.example).
Wynik: dane/surowe/spy/spy_5min_sip_<start>_<end>_pobrano_<data>.parquet (indeks w UTC).
"""
import os
import time
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import ROOT, SPY

load_dotenv(ROOT / ".env")  # klucze z pliku .env (zmienne środowiskowe mają pierwszeństwo)

URL = "https://data.alpaca.markets/v2/stocks/bars"


def get_headers():
    if not os.getenv("APCA_API_KEY_ID") or not os.getenv("APCA_API_SECRET_KEY"):
        raise RuntimeError(
            "Brakuje kluczy APCA_API_KEY_ID i APCA_API_SECRET_KEY. Uzupełnij plik .env (wzór: .env.example)."
        )
    return {
        "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
    }


def pobierz_swiece(symbol=SPY["symbol"], start=SPY["start"], end=SPY["end"], tf="5Min"):
    params = {
        "symbols": symbol,
        "timeframe": tf,
        "start": start,
        "end": end,
        "limit": 10000,
        "adjustment": "all",
        "feed": "sip",
        "sort": "asc",
    }
    wiersze = []

    while True:
        r = requests.get(URL, headers=get_headers(), params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(60)
            continue
        r.raise_for_status()
        dane = r.json()
        wiersze += (dane.get("bars") or {}).get(symbol, [])
        token = dane.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        time.sleep(0.35)

    if not wiersze:
        raise ValueError(f"Brak danych dla symbolu {symbol} w zakresie {start} - {end}.")

    df = pd.DataFrame(wiersze)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df.set_index("t").sort_index()


if __name__ == "__main__":
    spy = pobierz_swiece()
    SPY["surowe_dir"].mkdir(parents=True, exist_ok=True)
    rok_od, rok_do = SPY["start"][:4], SPY["end"][:4]
    output_path = SPY["surowe_dir"] / f"spy_5min_sip_{rok_od}_{rok_do}_pobrano_{date.today()}.parquet"
    spy.to_parquet(output_path)
    print(f"Pobrano {len(spy)} rekordów dla SPY. Zapisano do: {output_path}")
