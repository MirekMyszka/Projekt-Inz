# Pipeline danych

Dokument opisuje, jak pobierane i czyszczone są oba zbiory danych rzeczywistych: **SPY** (Alpaca) i **BTCUSDT** (Binance).
Wszystkie polecenia uruchamiamy z katalogu `Projekt-Inz`. Ścieżki i parametry są w `src/config.py`.

```
źródło ──► src/pobieranie/*.py ──► dane/surowe/<zbiór>/ ──► src/czyszczenie/*.py ──► dane/clean/<zbiór>/ ──► src/eksploracja/podglad.py ──► wyniki/wykresy/
```

Zasady wspólne:
- dane surowe zapisujemy dokładnie tak, jak przyszły (tylko konwersja do ramki z indeksem czasu UTC),
  z datą pobrania w nazwie; czyszczenie bierze **najnowszy** plik surowy,
- indeks `t` = **początek** świecy,
- brakujące świece **nie są uzupełniane** – oznaczamy je kolumnami i zapisujemy ich listę do CSV
  (wspólna funkcja `oznacz_luki` w `src/czyszczenie/wspolne.py`),
- klucze API tylko w `.env` (ignorowany przez Git).

---

## 1. SPY – Alpaca

### 1.1 Pobieranie – `python -m src.pobieranie.spy_alpaca`

- endpoint: `https://data.alpaca.markets/v2/stocks/bars`
- symbol `SPY`, interwał `5Min`, feed `sip`, `adjustment=all`, zakres 2016-01-01 – 2026-09-30
- klucze `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` z pliku `.env`
- stronicowanie po 10 000 rekordów (`next_page_token`), 0,35 s przerwy między zapytaniami,
  przy limicie 429 – 60 s czekania i ponowienie
- wynik: `dane/surowe/spy/spy_5min_sip_2016_2026_pobrano_<data>.parquet` (UTC, także handel poza sesją)

### 1.2 Czyszczenie – `python -m src.czyszczenie.spy`

1. Konwersja UTC → `America/New_York`.
2. Godziny sesji dla każdego dnia z kalendarza NYSE (`pandas_market_calendars`). Świeca zostaje, gdy
   `market_open <= t < market_close`:
   - zwykła sesja 09:30–16:00 → 78 świec (09:30 … 15:55),
   - skrócona sesja 09:30–13:00 (np. 3 lipca, piątek po Święcie Dziękczynienia, 24 grudnia) → 42 świece.

   Odrzucany jest handel przed- i posesyjny, w tym świece po 13:00 w dni skrócone.
3. Porównanie z oczekiwaną siatką (luki liczone w obrębie jednej sesji, noc nie jest luką). Kolumny:
   - `skrocona_sesja` – dzień ze skróconą sesją,
   - `luka_przed` – ile świec brakuje bezpośrednio przed daną świecą,
   - `dzien_niepelny` – w danym dniu brakuje co najmniej jednej świecy,
   - `wstrzymanie_handlu` – dzień z ogólnorynkowym wstrzymaniem handlu (circuit breaker): 9, 12, 16, 18 marca 2020.
4. Wynik: `dane/clean/spy/spy_5min_session_2016_2026.parquet` + `dane/clean/spy/spy_brakujace_swiece.csv`.

`luka_przed` liczy tylko braki **przed** istniejącą świecą tej samej sesji – braki na końcu sesji
(np. 2019-08-12 15:35–15:55) widać w `dzien_niepelny` i w pliku CSV, ale nie w `luka_przed`.

Wynik dla danych z 2026-10-13: 2701 dni (zgodnie z kalendarzem NYSE), 2674 dni po 78 świec, 21 skróconych sesji
po 42 świece, 14 brakujących świec w 6 dniach (8 – wstrzymania handlu w marcu 2020, 5 – koniec dnia 2019-08-12,
1 – 2016-02-02 11:20).

---

## 2. BTCUSDT – Binance

### 2.1 Pobieranie – `python -m src.pobieranie.btcusdt_binance`

- źródło: publiczne archiwum Binance, bez klucza API:
  `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-RRRR-MM.zip`
- zakres: 2017-08 (start pary na Binance, pierwsza świeca 2017-08-17) – 2026-09; inny zakres: `--start RRRR-MM --end RRRR-MM`
- każdy ZIP weryfikowany sumą SHA-256 z pliku `.CHECKSUM`; ZIP-y trzymane w `dane/cache/binance/` (poza Gitem) –
  przerwane pobieranie można wznowić, pobrane miesiące nie są ściągane ponownie
- 4 próby przy błędzie sieci; brak miesiąca w archiwum (404) kończy skrypt z czytelnym komunikatem

Format CSV w ZIP-ie (zwykle bez nagłówka – obsługujemy oba warianty):

```
open_time, open, high, low, close, volume, close_time, quote_volume, count,
taker_buy_base_volume, taker_buy_quote_volume, ignore
```

**Uwaga:** od 2025-01-01 `open_time` jest w **mikrosekundach** (16 cyfr), wcześniej w milisekundach (13 cyfr).
Skrypt sprowadza oba warianty do ms – bez tego daty od 2025 r. byłyby błędne.

Mapowanie kolumn (spójne z SPY):

| Wynik | Binance | Uwagi |
|---|---|---|
| `t` (indeks) | `open_time` | UTC, początek świecy |
| `o, h, l, c` | `open, high, low, close` | USDT |
| `v` | `volume` | wolumen w BTC |
| `n` | `count` | liczba transakcji |
| `qv` | `quote_volume` | obrót w USDT |
| `vw` | `quote_volume / volume` | VWAP; przy `v = 0` przyjmujemy `c` |

Wynik: `dane/surowe/btcusdt/btcusdt_5min_binance_2017_2026_pobrano_<data>.parquet`.

### 2.2 Czyszczenie – `python -m src.czyszczenie.btcusdt`

Rynek kryptowalut działa 24/7 – nie ma sesji ani kalendarza. Oczekiwana siatka to wszystkie znaczniki co 5 min
od pierwszej do ostatniej świecy (288 na dobę UTC).

1. Usunięcie duplikatów i sortowanie.
2. Wyrównanie znaczników czasu do siatki 5-min (`round`). Po przerwie technicznej 2018-02-08/09 Binance zapisał
   241 świec (2018-02-09 09:58:14.789 – 2018-02-10 05:58:14.789 UTC) przesuniętych o **194,789 s** względem siatki
   (minuty x3/x8 zamiast x0/x5). Każda jest przesuwana o 105,211 s do przodu – do przedziału, z którym pokrywa się
   w większości. Oznaczenie: `czas_skorygowany = True`. Jeśli wyrównanie utworzyłoby duplikaty, skrypt przerywa
   działanie błędem zamiast łączyć niejednoznaczne świece.
3. Kontrola spójności OHLC (raport, bez usuwania). Kolumny:
   - `luka_przed` – ile świec brakuje bezpośrednio przed daną świecą (szereg ciągły, także przez północ),
   - `dzien_niepelny` – w danym dniu (UTC) brakuje co najmniej jednej świecy,
   - `zerowy_wolumen` – świeca bez transakcji (`v == 0`, wtedy też `n == 0` i `o = h = l = c`),
   - `czas_skorygowany` – znacznik czasu został wyrównany do siatki (patrz p. 2).
4. Raport najdłuższych przerw (ciągłe bloki braków) – przerwy techniczne Binance.
5. Wynik: `dane/clean/btcusdt/btcusdt_5min_2017_2026.parquet` + `dane/clean/btcusdt/btcusdt_brakujace_swiece.csv`.

Weryfikacja danych z 2026-10-13 (niezależne przeliczenie):
- 957 865 świec, 2017-08-17 04:00 – 2026-09-30 23:55 UTC (siatka: 959 568), brak duplikatów i NaN,
- 3 332 dni, z czego 3 298 pełnych (288 świec); pierwszy dzień ma 240 świec (start notowań o 04:00 UTC – to nie luka),
- **1 703 brakujące świece w 33 przerwach** (każda przerwa w innym dniu); najdłuższe:
  2018-02-08 00:30 – 2018-02-09 09:55 (402 świece, ~33,5 h), 2018-06-26 i 2019-05-15 (po 120 świec, 10 h),
  2019-08-15 (96), 2018-07-04 (91), 2017-09-06 i 2018-11-14 (po 84),
- 0 świec z niespójnym OHLC, VWAP zawsze w `[l, h]`,
- 932 świece z zerowym wolumenem, wszystkie „płaskie” (`o = h = l = c`): 874 w 2017 r. (mała płynność na starcie rynku),
  pozostałe to krótkie bloki do 17 świec (2018–2023),
- 241 świec z `czas_skorygowany` (luty 2018).

---

## 3. Podgląd – `python -m src.eksploracja.podglad [spy|btcusdt]`

Dla każdego zbioru clean: kształt, zakres, pierwsze wiersze, statystyki opisowe i wykres ceny zamknięcia
(`wyniki/wykresy/<zbiór>_5min_close.png`; BTC w skali logarytmicznej).

## 4. Dalsze kroki

- log-zwroty (dla SPY tylko wewnątrz sesji), usunięcie sezonowości dziennej,
- wybór okresów do analizy rzeczywistej (np. marzec 2020),
- generator szeregów syntetycznych z anomaliami (`src/symulacje/`).
