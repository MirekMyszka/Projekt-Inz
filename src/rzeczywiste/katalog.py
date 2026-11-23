"""Katalog znanych zdarzeń w SPY i BTCUSDT - etykiety CZĘŚCIOWE do oceny metod na danych bez etykiet.

Źródło: docs/plan_metod.md, rozdz. 3 (przegląd danych z 2026-10-13, src/eksploracja/przeglad_anomalii.py, opis
przyczyn z wiadomości rynkowych). Typy jak w planie: A - punktowe (skrajna świeca), B - kolektywne (krótka seria),
C - zmiana poziomu ceny, D - reżim zmienności (wysokiej albo „nienaturalnego spokoju”), E - zaburzenie struktury
dobowej (FOMC, wygasanie opcji), F - wolumen. Daty: SPY - dni sesji (czas Nowego Jorku), BTC - doby UTC.

Zastrzeżenia (ograniczenia oceny):
- katalog powstał z przeglądu TYCH SAMYCH danych (największe zwroty, RV, wolumen) - zdarzenia są z definicji skrajne
  w prostych statystykach, więc „wykrycie” ich przez metody nie jest niezależnym testem; nie jest też kompletny
  (brak zdarzenia w katalogu ≠ brak anomalii) - dlatego zamiast precyzji liczymy recall@N i liczbę alarmów,
- typ E (struktura dobowa) jest widoczny w danych 5-min, a nie w szeregu dziennym - oceniany tylko w studiach przypadków.
"""
import pandas as pd

# (zbiór, typ, od, do, opis)
ZDARZENIA = [
    # ------------------------------------------------------------------ SPY
    ("SPY", "A", "2019-08-01", "2019-08-01", "zapowiedź nowych ceł na Chiny (skrajny ruch na tle spokojnego dnia)"),
    ("SPY", "A", "2020-03-12", "2020-03-12", "COVID - dzień ze wstrzymaniem handlu"),
    ("SPY", "A", "2025-04-07", "2025-04-07", "fałszywa informacja o wstrzymaniu ceł - skok i powrót w 10 min"),
    ("SPY", "A", "2025-10-10", "2025-10-10", "zapowiedź ceł - spadek −0,55% w 5 min (z lokalny −16,5), ten sam dzień co BTC"),
    ("SPY", "C", "2020-03-09", "2020-03-09", "luka overnight −7% (COVID)"),
    ("SPY", "C", "2020-03-16", "2020-03-16", "luka overnight −11,6% (COVID)"),
    ("SPY", "C", "2020-03-18", "2020-03-18", "luka overnight (COVID)"),
    ("SPY", "C", "2024-08-05", "2024-08-05", "luka overnight −4,1% (rozpad carry trade na jenie)"),
    ("SPY", "C", "2025-04-09", "2025-04-09", "ogłoszenie 90-dniowego zawieszenia ceł - przesunięcie poziomu o ok. +6%"),
    ("SPY", "D", "2017-05-01", "2017-11-30", "reżim nienaturalnego spokoju (RV 0,4× typowej) - anomalia „w dół”"),
    ("SPY", "D", "2018-12-01", "2018-12-31", "podwyższona zmienność (2–2,8×)"),
    ("SPY", "D", "2020-02-24", "2020-04-30", "COVID - zmienność do 6,4× typowej"),
    ("SPY", "D", "2022-05-01", "2022-06-30", "bessa 2022 - podwyższona zmienność"),
    ("SPY", "D", "2022-09-01", "2022-10-31", "bessa 2022 - podwyższona zmienność"),
    ("SPY", "D", "2025-04-01", "2025-04-30", "cła - zmienność 2× (dzień 2025-04-07: 12×)"),
    ("SPY", "F", "2016-06-24", "2016-06-24", "Brexit - wolumen 3,7×"),
    ("SPY", "F", "2018-02-05", "2018-02-06", "„Volmageddon” - wolumen 3,4× i 4,2× (rekord)"),
    ("SPY", "F", "2024-08-05", "2024-08-05", "wolumen 3,1×"),
    ("SPY", "F", "2025-04-04", "2025-04-09", "cła - wolumen 3,0–3,8×"),
    ("SPY", "E", "2021-06-16", "2021-06-16", "FOMC 14:00 - skok zmienności w środku dnia"),
    ("SPY", "E", "2022-09-21", "2022-09-21", "FOMC 14:00"),
    ("SPY", "E", "2023-12-13", "2023-12-13", "FOMC 14:00"),
    ("SPY", "E", "2024-12-18", "2024-12-18", "FOMC"),
    ("SPY", "E", "2025-10-29", "2025-10-29", "FOMC"),
    # ------------------------------------------------------------------ BTCUSDT
    ("BTC", "A", "2020-03-13", "2020-03-13", "odbicie po krachu COVID (+16,9% w 5 min)"),
    ("BTC", "A", "2021-09-07", "2021-09-07", "dzień wprowadzenia BTC jako waluty w Salwadorze (−10,4% w 5 min)"),
    ("BTC", "A", "2023-08-17", "2023-08-17", "flash crash w najspokojniejszym okresie (−9,1% w 5 min)"),
    ("BTC", "A", "2023-10-16", "2023-10-16", "fałszywa wiadomość o zatwierdzeniu ETF (+4,6%)"),
    ("BTC", "A", "2025-10-10", "2025-10-10", "kaskada likwidacji po zapowiedzi ceł (−7,6%)"),
    ("BTC", "B", "2018-10-15", "2018-10-15", "utrata parytetu USDT - anomalia notowania BTC/USDT"),
    ("BTC", "B", "2019-04-02", "2019-04-02", "seria wzrostów +3,5%, +3,2%, +3,1% w 45 min"),
    ("BTC", "B", "2021-05-19", "2021-05-19", "krach i odbicie: −7,6%, −10,2%, +9,0% w 25 min"),
    ("BTC", "C", "2017-09-14", "2017-09-14", "−19,2% w dobę (zakaz giełd w Chinach)"),
    ("BTC", "C", "2018-01-16", "2018-01-16", "−19,5% w dobę (krach po bańce)"),
    ("BTC", "C", "2019-10-25", "2019-10-25", "+16,8% w dobę"),
    ("BTC", "C", "2020-03-12", "2020-03-12", "−39,5% w dobę (COVID)"),
    ("BTC", "C", "2021-02-08", "2021-02-08", "+19,5% w dobę"),
    ("BTC", "C", "2022-06-13", "2022-06-13", "−15,4% w dobę (upadek Celsius)"),
    ("BTC", "D", "2017-08-17", "2018-03-31", "bańka i krach - zmienność 2–3,7×"),
    ("BTC", "D", "2018-10-01", "2018-10-31", "reżim spokoju (0,5×)"),
    ("BTC", "D", "2020-03-12", "2020-03-13", "COVID - dzienna RV 9–13×"),
    ("BTC", "D", "2021-01-01", "2021-01-31", "podwyższona zmienność 2,3×"),
    ("BTC", "D", "2021-05-01", "2021-05-31", "podwyższona zmienność 2,5×"),
    ("BTC", "D", "2022-12-01", "2022-12-31", "reżim spokoju (0,5×)"),
    ("BTC", "D", "2023-07-01", "2023-08-31", "reżim spokoju (0,5×), tło flash crashu 2023-08-17"),
    ("BTC", "F", "2020-03-13", "2020-03-13", "wolumen 5,2×"),
    ("BTC", "F", "2021-05-19", "2021-05-19", "wolumen 4,1×"),
    ("BTC", "F", "2024-08-05", "2024-08-05", "wolumen 5,9× (rekord) - to samo zdarzenie co w SPY"),
    ("BTC", "F", "2026-02-05", "2026-02-06", "wolumen 5,0× i 4,3×"),
]
OPIS_TYPU = {"A": "punktowe", "B": "kolektywne", "C": "zmiana poziomu", "D": "reżim zmienności", "E": "struktura dobowa",
             "F": "wolumen"}
# Wspólne zdarzenia obu zbiorów - studia przypadków na danych 5-min (plan_metod.md, rozdz. 3)
STUDIA = [("marzec 2020 (COVID)", "2020-03-09", "2020-03-20"), ("2024-08-05 (carry trade)", "2024-08-02", "2024-08-07"),
          ("kwiecień 2025 (cła)", "2025-04-03", "2025-04-10"), ("2025-10-10 (cła, likwidacje)", "2025-10-09", "2025-10-13")]


def katalog() -> pd.DataFrame:
    k = pd.DataFrame(ZDARZENIA, columns=["zbior", "typ", "od", "do", "opis"])
    k["od"], k["do"] = pd.to_datetime(k["od"]).dt.date, pd.to_datetime(k["do"]).dt.date
    return k
