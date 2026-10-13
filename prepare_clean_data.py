"""Czyszczenie danych SPY 5-min do godzin rzeczywistej sesji NYSE.

Kroki:
1. Konwersja indeksu z UTC do America/New_York.
2. Filtrowanie do godzin sesji z kalendarza NYSE (pandas_market_calendars):
   - zwykła sesja 09:30-16:00 -> 78 świec (09:30 ... 15:55),
   - skrócona sesja 09:30-13:00 -> 42 świece (09:30 ... 12:55).
   Indeks t to początek świecy, więc świeca jest w sesji, gdy open <= t < close.
3. Oznaczenie brakujących świec (np. wstrzymania handlu - circuit breaker w marcu 2020).
   Brakujące świece NIE są uzupełniane; dodawane są kolumny:
   - skrocona_sesja  : bool, dzień ze skróconą sesją (zamknięcie przed 16:00),
   - luka_przed      : int, ile oczekiwanych świec brakuje bezpośrednio przed tą świecą,
   - dzien_niepelny  : bool, w danym dniu brakuje co najmniej jednej świecy,
   - wstrzymanie_handlu : bool, dzień z circuit breakerem (9, 12, 16, 18 marca 2020).
   Lista brakujących znaczników czasu zapisywana jest do dane/clean/brakujace_swiece.csv.
"""
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "dane" / "surowe" / "spy_5min_sip_2016_2026_pobrano_2026-10-13.parquet"
OUT_PATH = BASE_DIR / "dane" / "clean" / "spy_5min_session_2016_2026.parquet"
MISSING_PATH = BASE_DIR / "dane" / "clean" / "brakujace_swiece.csv"
TZ = "America/New_York"
FREQ = "5min"

# Dni z ogólnorynkowym wstrzymaniem handlu (Market-Wide Circuit Breaker, poziom 1: spadek S&P 500 o 7%).
# Handel wstrzymano na 15 min krótko po otwarciu, stąd brakujące świece 5-min.
CIRCUIT_BREAKER_DAYS = pd.to_datetime(["2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"]).date


def nyse_schedule(start, end) -> pd.DataFrame:
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    sched = sched[["market_open", "market_close"]].apply(lambda s: s.dt.tz_convert(TZ))
    sched.index = sched.index.date
    return sched


def expected_grid(sched: pd.DataFrame) -> pd.DatetimeIndex:
    parts = [
        pd.date_range(o, c, freq=FREQ, inclusive="left")
        for o, c in zip(sched["market_open"], sched["market_close"])
    ]
    return parts[0].append(parts[1:])


def prepare_clean_data():
    raw = pd.read_parquet(RAW_PATH)
    # zakres kalendarza wg dat UTC, żeby nie łapać dnia poprzedzającego (00:00 UTC = 19:00 NY dnia wcześniej)
    sched = nyse_schedule(raw.index.min().date(), raw.index.max().date())
    spy = raw.tz_convert(TZ)

    # 1) tylko świece w godzinach sesji danego dnia
    day = pd.Index(spy.index.date)
    opens = pd.Series(sched["market_open"]).reindex(day).to_numpy()
    closes = pd.Series(sched["market_close"]).reindex(day).to_numpy()
    in_session = pd.notna(opens) & (spy.index >= opens) & (spy.index < closes)
    sesja = spy[in_session].copy()

    # 2) oznaczenie skróconych sesji
    full_close = sched["market_close"].dt.hour == 16
    short_days = set(sched.index[~full_close])
    sesja_day = pd.Index(sesja.index.date)
    sesja["skrocona_sesja"] = sesja_day.isin(short_days)

    # 3) brakujące świece względem oczekiwanej siatki
    grid = expected_grid(sched)
    missing = grid.difference(sesja.index)
    pos = grid.get_indexer(sesja.index)
    gap = pd.Series(pos, index=sesja.index).groupby(sesja_day).diff()
    first_in_day = pd.Series(pos, index=sesja.index).groupby(sesja_day).transform("min")
    day_start_pos = pd.Series(grid.get_indexer(
        pd.DatetimeIndex(sched.loc[sesja_day, "market_open"].to_numpy()).tz_convert(TZ)
    ), index=sesja.index)
    luka = (gap - 1).fillna(first_in_day - day_start_pos).astype(int)
    sesja["luka_przed"] = luka
    incomplete_days = set(pd.Index(missing.date))
    sesja["dzien_niepelny"] = sesja_day.isin(incomplete_days)
    sesja["wstrzymanie_handlu"] = sesja_day.isin(CIRCUIT_BREAKER_DAYS)

    # raport
    liczba = sesja.groupby(sesja_day).size()
    print("Dni sesyjnych w danych:", len(liczba), "| dni w kalendarzu NYSE:", len(sched))
    print("Skrócone sesje:", int(sesja.groupby(sesja_day)["skrocona_sesja"].first().sum()))
    print("Rozkład liczby świec na dzień:")
    print(liczba.value_counts().sort_index().to_string())
    print(f"Brakujące świece: {len(missing)} w {len(incomplete_days)} dniach")
    if len(missing):
        print(pd.Series(missing.date).value_counts().sort_index().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sesja.to_parquet(OUT_PATH)
    pd.DataFrame({"t": missing}).to_csv(MISSING_PATH, index=False)
    print(f"Zapisano dane sesyjne do: {OUT_PATH}")
    print(f"Zapisano listę brakujących świec do: {MISSING_PATH}")


if __name__ == "__main__":
    prepare_clean_data()
