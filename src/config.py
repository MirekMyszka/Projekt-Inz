"""Wspólna konfiguracja: ścieżki do danych i wyników, parametry zbiorów danych.

Wszystkie skrypty uruchamiamy z katalogu głównego repozytorium (Projekt-Inz), np.:
    python -m src.pobieranie.btcusdt_binance
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DANE = ROOT / "dane"
SUROWE = DANE / "surowe"     # dane dokładnie tak, jak przyszły ze źródła
CLEAN = DANE / "clean"       # dane po czyszczeniu, gotowe do analizy
CACHE = DANE / "cache"       # pliki pośrednie (np. ZIP-y z Binance) - poza Gitem

WYNIKI = ROOT / "wyniki"
WYKRESY = WYNIKI / "wykresy"

FREQ = "5min"

# --- SPY (Alpaca) ---
SPY = {
    "symbol": "SPY",
    "start": "2016-01-01",
    "end": "2026-09-30",
    "surowe_dir": SUROWE / "spy",
    "surowe_wzorzec": "spy_5min_sip_*.parquet",
    "clean": CLEAN / "spy" / "spy_5min_session_2016_2026.parquet",
    "brakujace": CLEAN / "spy" / "spy_brakujace_swiece.csv",
    "tz": "America/New_York",
}

# --- BTCUSDT (Binance spot) ---
BTCUSDT = {
    "symbol": "BTCUSDT",
    "start": "2017-08",          # pierwszy miesiąc notowań BTCUSDT na Binance (od 2017-08-17)
    "end": "2026-09",            # ostatni pełny miesiąc (spójnie z SPY)
    "interwal": "5m",
    "surowe_dir": SUROWE / "btcusdt",
    "surowe_wzorzec": "btcusdt_5min_binance_*.parquet",
    "clean": CLEAN / "btcusdt" / "btcusdt_5min_2017_2026.parquet",
    "brakujace": CLEAN / "btcusdt" / "btcusdt_brakujace_swiece.csv",
    "cache": CACHE / "binance",
    "tz": "UTC",
}


def najnowszy_plik(katalog: Path, wzorzec: str) -> Path:
    """Zwraca najnowszy (alfabetycznie = wg daty pobrania w nazwie) plik pasujący do wzorca."""
    pliki = sorted(katalog.glob(wzorzec))
    if not pliki:
        raise FileNotFoundError(f"Brak pliku {wzorzec} w {katalog}. Najpierw uruchom pobieranie.")
    return pliki[-1]
