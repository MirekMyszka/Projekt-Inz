"""Czyszczenie danych SPY 5-min do godzin rzeczywistej sesji NYSE.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.czyszczenie.spy

Kroki:
1. Konwersja indeksu z UTC do America/New_York.
2. Filtrowanie do godzin sesji z kalendarza NYSE (pandas_market_calendars):
   - zwykła sesja 09:30-16:00 -> 78 świec (09:30 ... 15:55),
   - skrócona sesja 09:30-13:00 -> 42 świece (09:30 ... 12:55).
   Indeks t to początek świecy, więc świeca jest w sesji, gdy open <= t < close.
3. Oznaczenie brakujących świec (np. wstrzymania handlu - circuit breaker w marcu 2020).
   Brakujące świece NIE są uzupełniane; dodawane są kolumny:
   - skrocona_sesja     : bool, dzień ze skróconą sesją (zamknięcie przed 16:00),
   - luka_przed         : int, ile oczekiwanych świec brakuje bezpośrednio przed tą świecą,
   - dzien_niepelny     : bool, w danym dniu brakuje co najmniej jednej świecy,
   - wstrzymanie_handlu : bool, dzień z circuit breakerem (9, 12, 16, 18 marca 2020).
   Lista brakujących znaczników czasu: dane/clean/spy/spy_brakujace_swiece.csv.
"""
import pandas as pd
import pandas_market_calendars as mcal

from src.config import FREQ, SPY, najnowszy_plik
from src.czyszczenie.wspolne import oznacz_luki, raport_luk

TZ = SPY["tz"]

# Dni z ogólnorynkowym wstrzymaniem handlu (Market-Wide Circuit Breaker, poziom 1: spadek S&P 500 o 7%).
# Handel wstrzymano na 15 min, stąd brakujące świece 5-min.
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
    raw_path = najnowszy_plik(SPY["surowe_dir"], SPY["surowe_wzorzec"])
    print("Plik surowy:", raw_path.name)
    raw = pd.read_parquet(raw_path)
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

    # 3) brakujące świece względem oczekiwanej siatki (luki liczone w obrębie jednej sesji)
    sesja, missing = oznacz_luki(sesja, expected_grid(sched), segment=lambda idx: idx.date)
    sesja["wstrzymanie_handlu"] = sesja_day.isin(CIRCUIT_BREAKER_DAYS)

    # raport
    liczba = sesja.groupby(sesja_day).size()
    print("Dni sesyjnych w danych:", len(liczba), "| dni w kalendarzu NYSE:", len(sched))
    print("Skrócone sesje:", int(sesja.groupby(sesja_day)["skrocona_sesja"].first().sum()))
    print("Rozkład liczby świec na dzień:")
    print(liczba.value_counts().sort_index().to_string())
    raport_luk(missing)

    SPY["clean"].parent.mkdir(parents=True, exist_ok=True)
    sesja.to_parquet(SPY["clean"])
    pd.DataFrame({"t": missing}).to_csv(SPY["brakujace"], index=False, lineterminator="\n")
    print(f"Zapisano dane sesyjne do: {SPY['clean']}")
    print(f"Zapisano listę brakujących świec do: {SPY['brakujace']}")


if __name__ == "__main__":
    prepare_clean_data()
