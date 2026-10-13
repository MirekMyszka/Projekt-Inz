"""Pobieranie świec 5-min BTCUSDT (Binance spot) z publicznego archiwum data.binance.vision.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.pobieranie.btcusdt_binance
    python -m src.pobieranie.btcusdt_binance --start 2024-01 --end 2024-12   # inny zakres

- Nie wymaga klucza API ani konta.
- Jeden plik ZIP = jeden miesiąc świec. Każdy ZIP jest weryfikowany sumą SHA-256 z pliku .CHECKSUM.
- Pobrane ZIP-y są trzymane w dane/cache/binance/ (poza Gitem) - ponowne uruchomienie pobiera tylko brakujące.
- Wynik: dane/surowe/btcusdt/btcusdt_5min_binance_<rok_od>_<rok_do>_pobrano_<data>.parquet (indeks w UTC).

Format CSV z archiwum (bez nagłówka; w części plików nagłówek bywa - obsługujemy oba warianty):
    open_time, open, high, low, close, volume, close_time, quote_volume, count,
    taker_buy_base_volume, taker_buy_quote_volume, ignore
UWAGA: od 2025-01-01 znaczniki czasu w archiwum spot są w MIKROsekundach (wcześniej milisekundy).

Kolumny wyniku (spójne z SPY): o, h, l, c, v (wolumen w BTC), n (liczba transakcji),
vw (VWAP = obrót w USDT / wolumen w BTC) oraz qv (obrót w USDT).
"""
import argparse
import hashlib
import io
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.config import BTCUSDT

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interwal}/{symbol}-{interwal}-{miesiac}.zip"
KOLUMNY = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def miesiace(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _get(sesja: requests.Session, url: str, proby: int = 4) -> requests.Response:
    for i in range(proby):
        try:
            r = sesja.get(url, timeout=60)
            if r.status_code == 404:
                raise FileNotFoundError(f"Brak pliku w archiwum Binance (404): {url}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if i == proby - 1:
                raise
            print(f"  błąd ({e}), ponawiam za {5 * (i + 1)} s...")
            time.sleep(5 * (i + 1))


def pobierz_zip(sesja: requests.Session, miesiac: str) -> bytes:
    """Zwraca zawartość ZIP-a dla miesiąca (z cache albo z sieci), zweryfikowaną sumą SHA-256."""
    url = BASE_URL.format(symbol=BTCUSDT["symbol"], interwal=BTCUSDT["interwal"], miesiac=miesiac)
    cache_dir = BTCUSDT["cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / url.rsplit("/", 1)[-1]
    sum_path = zip_path.with_name(zip_path.name + ".CHECKSUM")

    if zip_path.exists() and sum_path.exists():
        dane = zip_path.read_bytes()
        if hashlib.sha256(dane).hexdigest() == sum_path.read_text().split()[0]:
            return dane
        print(f"  {zip_path.name}: suma kontrolna się nie zgadza, pobieram ponownie")

    oczekiwana = _get(sesja, url + ".CHECKSUM").text.split()[0]
    dane = _get(sesja, url).content
    faktyczna = hashlib.sha256(dane).hexdigest()
    if faktyczna != oczekiwana:
        raise ValueError(f"Zła suma SHA-256 dla {zip_path.name}: {faktyczna} != {oczekiwana}")
    zip_path.write_bytes(dane)
    sum_path.write_text(f"{oczekiwana}  {zip_path.name}\n")
    return dane


def wczytaj_zip(dane: bytes) -> pd.DataFrame:
    """Parsuje miesięczny ZIP z archiwum Binance do surowej ramki z kolumnami KOLUMNY."""
    with zipfile.ZipFile(io.BytesIO(dane)) as zf:
        nazwa = [n for n in zf.namelist() if n.endswith(".csv")][0]
        tekst = zf.read(nazwa)
    df = pd.read_csv(io.BytesIO(tekst), header=None, names=KOLUMNY, dtype=str)
    if not df.iloc[0, 0].strip().isdigit():  # plik z nagłówkiem
        df = df.iloc[1:]
    return df


def przetworz(surowe: pd.DataFrame) -> pd.DataFrame:
    """Surowe kolumny Binance -> o, h, l, c, v, n, vw, qv z indeksem t (UTC, początek świecy)."""
    open_time = surowe["open_time"].astype("int64").to_numpy()
    # od 2025 r. mikrosekundy (16 cyfr), wcześniej milisekundy (13 cyfr) -> sprowadzamy do ms
    open_time_ms = np.where(open_time >= 10**14, open_time // 1000, open_time)

    df = pd.DataFrame(
        {
            "o": surowe["open"].astype(float).to_numpy(),
            "h": surowe["high"].astype(float).to_numpy(),
            "l": surowe["low"].astype(float).to_numpy(),
            "c": surowe["close"].astype(float).to_numpy(),
            "v": surowe["volume"].astype(float).to_numpy(),
            "n": surowe["count"].astype("int64").to_numpy(),
            "qv": surowe["quote_volume"].astype(float).to_numpy(),
        },
        index=pd.DatetimeIndex(pd.to_datetime(open_time_ms, unit="ms", utc=True), name="t"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        df["vw"] = np.where(df["v"] > 0, df["qv"] / df["v"], df["c"])
    df = df[["o", "h", "l", "c", "v", "n", "vw", "qv"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def pobierz_swiece(start: str = BTCUSDT["start"], end: str = BTCUSDT["end"]) -> pd.DataFrame:
    lista = miesiace(start, end)
    print(f"Pobieranie {BTCUSDT['symbol']} {BTCUSDT['interwal']}, liczba miesięcy: {len(lista)} ({lista[0]} - {lista[-1]})")
    czesci = []
    with requests.Session() as sesja:
        for i, m in enumerate(lista, 1):
            czesci.append(wczytaj_zip(pobierz_zip(sesja, m)))
            if i % 12 == 0 or i == len(lista):
                print(f"  {i}/{len(lista)} ({m})")
    return przetworz(pd.concat(czesci, ignore_index=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=BTCUSDT["start"], help="pierwszy miesiąc RRRR-MM")
    parser.add_argument("--end", default=BTCUSDT["end"], help="ostatni miesiąc RRRR-MM")
    args = parser.parse_args()

    btc = pobierz_swiece(args.start, args.end)
    BTCUSDT["surowe_dir"].mkdir(parents=True, exist_ok=True)
    out = BTCUSDT["surowe_dir"] / f"btcusdt_5min_binance_{args.start[:4]}_{args.end[:4]}_pobrano_{date.today()}.parquet"
    btc.to_parquet(out)
    print(f"Pobrano {len(btc)} świec ({btc.index.min()} - {btc.index.max()}). Zapisano do: {out}")
