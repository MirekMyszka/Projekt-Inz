# Pipeline danych dla projektu SPY

## Cel

Ten dokument opisuje, jak pobierane są dane rynkowe z Alpaca, jak są filtrowane i jak są przygotowywane do dalszej analizy lub modelowania.

## 1. Źródło danych

Dane są pobierane z API Alpaca:

- endpoint: `https://data.alpaca.markets/v2/stocks/bars`
- symbol: `SPY`
- interwał: `5Min`
- feed: `sip`
- korekta: `adjustment=all`

Dane są pobierane dla okresu:

- start: `2016-01-01`
- end: `2026-09-30`

## 2. Lokalizacja danych

Surowe dane są zapisywane w katalogu:

- `dane/surowe/`

Aktualnie plik ma nazwę typu:

- `spy_5min_sip_2016_2026_pobrano_2026-10-13.parquet`

## 3. Standard pobierania

Skrypt pobiera dane w blokach po maksymalnie 10000 rekordów na żądanie. Jeśli API zwróci limit 429, skrypt czeka 60 sekund i ponawia żądanie.

Dla każdej odpowiedzi:

- pobierane są bary (świece) dla symbolu SPY,
- jeżeli istnieje `next_page_token`, pobieranie jest kontynuowane,
- po każdym kolejnym żądaniu wstawiany jest mały odstęp czasu (0.35 s), aby nie przekroczyć limitu zapytań.

## 4. Ustawianie strefy czasowej

Surowe dane są zapisywane w UTC. Dopiero w skrypcie `prepare_clean_data.py` są konwertowane do strefy czasowej giełdowej:

- `America/New_York`

Dzięki temu indeks timestampów odpowiada czasowi sesji handlowej na rynku USA.

## 5. Filtrowanie sesji handlowej (kalendarz NYSE)

Godziny sesji dla każdego dnia pobierane są z kalendarza NYSE (`pandas_market_calendars`). Świeca (indeks = jej początek) zostaje, gdy:

```
market_open <= t < market_close
```

- zwykła sesja 09:30–16:00 → 78 świec (09:30 … 15:55),
- skrócona sesja 09:30–13:00 (np. 3 lipca, piątek po Święcie Dziękczynienia, 24 grudnia) → 42 świece (09:30 … 12:55).

Odrzucany jest handel przed- i posesyjny, w tym świece po 13:00 w dni skrócone. Wynik zapisywany jest do `dane/clean/spy_5min_session_2016_2026.parquet`.

## 6. Kontrola kompletności i oznaczanie luk

Dla każdego dnia budowana jest oczekiwana siatka 5-min z kalendarza i porównywana z danymi. Brakujące świece **nie są uzupełniane**, tylko oznaczane kolumnami:

- `skrocona_sesja` – dzień ze skróconą sesją,
- `luka_przed` – ile świec brakuje bezpośrednio przed daną świecą,
- `dzien_niepelny` – w danym dniu brakuje co najmniej jednej świecy,
- `wstrzymanie_handlu` – dzień z ogólnorynkowym wstrzymaniem handlu (circuit breaker): 9, 12, 16, 18 marca 2020.

Lista brakujących znaczników czasu: `dane/clean/brakujace_swiece.csv`.

Wynik dla danych z 2026-10-13: 2701 dni (zgodnie z kalendarzem NYSE), 2674 dni po 78 świec, 21 skróconych sesji po 42 świece, 14 brakujących świec w 6 dniach (8 z nich to wstrzymania handlu w marcu 2020, 5 to koniec dnia 2019-08-12, 1 to 2016-02-02 11:20).

## 7. Podgląd i wizualizacja danych

Dodatkowo przygotowany jest skrypt `preview_data.py`, który:

- odczytuje dane z pliku Parquet,
- pokazuje kształt ramki danych i podstawowe statystyki,
- wyświetla pierwsze wiersze,
- zapisuje do `plots/spy_5min_close.png` wykres szeregu czasowego ceny zamknięcia (`close`, kolumna `c`).

Wykres jest ważny, ponieważ pozwala szybko zauważyć:

- trend i ewolucję ceny,
- wyraźne skoki cenowe,
- luki w danych,
- anomalie lub niepełne okresy sesji.

Przykładowy fragment wizualizacji:

```python
if __name__ == "__main__":
    df = pd.read_parquet(CLEAN_PATH)
    df["c"].plot(figsize=(14, 6), title="SPY 5Min Close")
    plt.show()
```

## 8. Dalsze kroki

Po tej wstępnej walidacji dane można:

- oczyścić z braków i niepełnych dni,
- dodać kolumny z datą, dniem tygodnia, godziną,
- zbudować cechy techniczne i statystyczne,
- przygotować dataset do modelowania lub backtestu.

## 9. Zasady projektu

- pliki z danymi surowymi są przechowywane lokalnie,
- klucze API nie trafiają do repozytorium,
- plik `.env` jest ignorowany przez Git,
- dokumentacja jest zapisywana w katalogu `docs/`.
