# Etap 2 – dane rzeczywiste: tabele (generowane przez src/rzeczywiste/wykresy_rzeczywiste.py)

## NYC Taxi – wykryte okna NAB i dni z fałszywym alarmem w okresie oceny; próg z okresu kalibracyjnego (alarm w α dni kalibracji)

| metoda | α = 2% | α = 5% | α = 10% | AUC | AP |
|---|---|---|---|---|---|
| STL + EWMA | 4/5 · 1% | 5/5 · 5% | 5/5 · 15% | 0,87 | 0,78 |
| PELT (reszty STL) | 2/5 · 0% | 4/5 · 0% | 4/5 · 28% | 0,77 | 0,64 |
| Matrix Profile z normalizacją | 4/5 · 3% | 5/5 · 3% | 5/5 · 6% | 0,87 | 0,70 |
| cechy cykli + kNN (5 cech) | 2/5 · 0% | 5/5 · 3% | 5/5 · 5% | 0,90 | 0,84 |
| cechy cykli + kNN (3 cechy) | 2/5 · 0% | 5/5 · 3% | 5/5 · 6% | 0,89 | 0,82 |
| Isolation Forest (5 cech) | 3/5 · 0% | 5/5 · 3% | 5/5 · 8% | 0,88 | 0,82 |
| Isolation Forest (3 cechy) | 3/5 · 0% | 5/5 · 3% | 5/5 · 4% | 0,87 | 0,81 |
| autoenkoder | 3/5 · 4% | 3/5 · 4% | 3/5 · 10% | 0,79 | 0,54 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 3/5 · 0% | 5/5 · 3% | 5/5 · 10% | 0,88 | 0,82 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 3/5 · 0% | 5/5 · 3% | 5/5 · 4% | 0,88 | 0,82 |
| autoenkoder (uczony na okresie kalibracyjnym) | 4/5 · 5% | 4/5 · 10% | 4/5 · 15% | 0,74 | 0,52 |
| Matrix Profile bez normalizacji | 5/5 · 3% | 5/5 · 4% | 5/5 · 6% | 0,93 | 0,82 |
| CUSUM z S1 (bez modelu cyklu) | 0/5 · 0% | 0/5 · 0% | 0/5 · 0% | 0,50 | 0,21 |

## NYC Taxi – wykryte okna NAB i dni z fałszywym alarmem w okresie oceny; równy budżet (alarm w α dni bez anomalii okresu oceny)

| metoda | α = 2% | α = 5% | α = 10% | AUC | AP |
|---|---|---|---|---|---|
| STL + EWMA | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,87 | 0,78 |
| PELT (reszty STL) | 4/5 · 1% | 4/5 · 4% | 4/5 · 5% | 0,77 | 0,64 |
| Matrix Profile z normalizacją | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,87 | 0,70 |
| cechy cykli + kNN (5 cech) | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,90 | 0,84 |
| cechy cykli + kNN (3 cechy) | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,89 | 0,82 |
| Isolation Forest (5 cech) | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,88 | 0,82 |
| Isolation Forest (3 cechy) | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,87 | 0,81 |
| autoenkoder | 3/5 · 3% | 3/5 · 5% | 3/5 · 10% | 0,79 | 0,54 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,88 | 0,82 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,88 | 0,82 |
| autoenkoder (uczony na okresie kalibracyjnym) | 3/5 · 3% | 4/5 · 5% | 4/5 · 10% | 0,74 | 0,52 |
| Matrix Profile bez normalizacji | 5/5 · 3% | 5/5 · 5% | 5/5 · 10% | 0,93 | 0,82 |
| CUSUM z S1 (bez modelu cyklu) | 0/5 · 0% | 0/5 · 0% | 0/5 · 0% | 0,50 | 0,21 |

## NYC Taxi – AUC dla każdego okna

| metoda | maraton nowojorski | Święto Dziękczynienia | Boże Narodzenie | Nowy Rok | śnieżyca (zakaz ruchu) |
|---|---|---|---|---|---|
| STL + EWMA | 0,65 | 0,96 | 0,99 | 0,99 | 0,76 |
| PELT (reszty STL) | 0,35 | 0,93 | 0,99 | 0,99 | 0,60 |
| Matrix Profile z normalizacją | 0,80 | 0,87 | 0,87 | 0,95 | 0,87 |
| cechy cykli + kNN (5 cech) | 0,68 | 0,98 | 0,99 | 0,99 | 0,86 |
| cechy cykli + kNN (3 cechy) | 0,66 | 0,94 | 0,99 | 0,99 | 0,86 |
| Isolation Forest (5 cech) | 0,65 | 0,97 | 0,99 | 0,99 | 0,81 |
| Isolation Forest (3 cechy) | 0,63 | 0,91 | 0,98 | 1,00 | 0,86 |
| autoenkoder | 0,59 | 0,79 | 0,94 | 0,82 | 0,82 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 0,63 | 0,96 | 0,98 | 0,99 | 0,82 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 0,64 | 0,89 | 0,98 | 1,00 | 0,86 |
| autoenkoder (uczony na okresie kalibracyjnym) | 0,60 | 0,71 | 0,91 | 0,68 | 0,80 |
| Matrix Profile bez normalizacji | 0,78 | 0,96 | 0,98 | 0,97 | 0,92 |
| CUSUM z S1 (bez modelu cyklu) | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 |

## NYC Taxi – święta nieoznaczone w NAB: miejsce dnia w rankingu dziennych maksimów wyniku (w swoim okresie: kalibracja 105 dni, ocena 110 dni)

| dzień | STL + EWMA | PELT (reszty STL) | Matrix Profile z normalizacją | cechy cykli + kNN (5 cech) | cechy cykli + kNN (3 cechy) | Isolation Forest (5 cech) | Isolation Forest (3 cechy) | autoenkoder | Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | autoenkoder (uczony na okresie kalibracyjnym) | Matrix Profile bez normalizacji | CUSUM z S1 (bez modelu cyklu) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2014-07-04 Dzień Niepodległości (kalibracja) | 2 | 1 | 2 | 4 | 3 | 3 | 3 | 16 | 3 | 3 | 17 | 2 | 1 |
| 2014-09-01 Labor Day (kalibracja) | 1 | 5 | 4 | 1 | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 7 | 1 |
| 2014-10-13 Columbus Day (kalibracja) | 5 | 9 | 8 | 6 | 7 | 8 | 7 | 31 | 7 | 7 | 37 | 4 | 1 |
| 2014-11-11 Dzień Weterana (ocena) | 44 | 45 | 52 | 74 | 66 | 78 | 61 | 100 | 69 | 57 | 110 | 48 | 1 |
| 2015-01-19 Dzień Martina Luthera Kinga (ocena) | 11 | 45 | 9 | 18 | 17 | 16 | 17 | 24 | 15 | 15 | 23 | 11 | 1 |

## SPY – szereg: zmiennosc; recall@N (N = 1 / 2 / 5% dni) według typu zdarzenia; wykrycie przy progu z kalibracji (α = 2%) w okresie oceny i liczba dni z alarmem na rok

| metoda | A punktowe (n = 4) | C zmiana poziomu (n = 5) | D reżim zmienności (n = 6) | wykryte w okresie oceny (próg α = 2%) | dni z alarmem na rok (ocena) |
|---|---|---|---|---|---|
| reszty AR + CUSUM | 0% / 0% / 0% | 0% / 0% / 0% | 0% / 0% / 0% | 0% | 31,5 |
| PELT (koszt AR) | 0% / 0% / 0% | 0% / 0% / 0% | 0% / 0% / 0% | 0% | 0,0 |
| Matrix Profile (reszty AR) | 25% / 50% / 100% | 80% / 80% / 80% | 33% / 33% / 33% | 75% | 9,7 |
| cechy okien + kNN (5 cech) | 25% / 50% / 50% | 60% / 80% / 80% | 33% / 67% / 100% | 83% | 10,4 |
| cechy okien + kNN (3 cechy) | 50% / 50% / 50% | 80% / 80% / 80% | 50% / 50% / 100% | 83% | 10,7 |
| Isolation Forest (5 cech) | 50% / 50% / 75% | 80% / 80% / 80% | 33% / 50% / 83% | 83% | 14,7 |
| Isolation Forest (3 cechy) | 50% / 50% / 50% | 80% / 80% / 80% | 33% / 67% / 100% | 83% | 15,2 |
| One-Class SVM (5 cech) | 50% / 50% / 50% | 80% / 80% / 80% | 33% / 33% / 83% | 75% | 9,8 |
| One-Class SVM (3 cechy) | 50% / 50% / 50% | 80% / 80% / 80% | 50% / 50% / 100% | 83% | 9,5 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 50% / 50% / 75% | 80% / 80% / 80% | 50% / 67% / 83% | 83% | 11,4 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 50% / 50% / 50% | 80% / 80% / 80% | 67% / 67% / 100% | 75% | 12,9 |
| One-Class SVM (5 cech, uczony na okresie kalibracyjnym) | 25% / 50% / 50% | 60% / 80% / 80% | 17% / 50% / 83% | 75% | 8,3 |
| One-Class SVM (3 cechy, uczony na okresie kalibracyjnym) | 50% / 50% / 50% | 80% / 80% / 80% | 50% / 67% / 67% | 75% | 7,9 |
| CUSUM z S1 (bez modelu AR) | 25% / 25% / 25% | 60% / 60% / 60% | 17% / 17% / 17% | 58% | 116,1 |

## SPY – szereg: wolumen; recall@N (N = 1 / 2 / 5% dni) według typu zdarzenia; wykrycie przy progu z kalibracji (α = 2%) w okresie oceny i liczba dni z alarmem na rok

| metoda | F wolumen (n = 4) | wykryte w okresie oceny (próg α = 2%) | dni z alarmem na rok (ocena) |
|---|---|---|---|
| reszty AR + CUSUM | 0% / 0% / 0% | 0% | 55,2 |
| PELT (koszt AR) | 0% / 0% / 0% | 100% | 185,2 |
| Matrix Profile (reszty AR) | 50% / 50% / 50% | 50% | 1,9 |
| cechy okien + kNN (5 cech) | 25% / 25% / 75% | 0% | 5,9 |
| cechy okien + kNN (3 cechy) | 0% / 50% / 50% | 0% | 4,9 |
| Isolation Forest (5 cech) | 50% / 75% / 75% | 50% | 2,8 |
| Isolation Forest (3 cechy) | 25% / 75% / 75% | 50% | 4,6 |
| One-Class SVM (5 cech) | 50% / 50% / 50% | 0% | 5,6 |
| One-Class SVM (3 cechy) | 25% / 50% / 50% | 0% | 4,5 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 50% / 75% / 75% | 50% | 0,9 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 50% / 75% / 75% | 50% | 1,6 |
| One-Class SVM (5 cech, uczony na okresie kalibracyjnym) | 25% / 50% / 50% | 0% | 4,3 |
| One-Class SVM (3 cechy, uczony na okresie kalibracyjnym) | 0% / 50% / 75% | 0% | 4,2 |
| CUSUM z S1 (bez modelu AR) | 0% / 0% / 25% | 0% | 0,0 |

## SPY – zgodność metod (indeks Jaccarda zbiorów 2% najwyższych dni; bez CUSUM z S1)

| szereg | mediana | min | max |
|---|---|---|---|
| wolumen | 0,33 | 0,00 | 0,74 |
| zmiennosc | 0,30 | 0,00 | 0,74 |

## SPY – dni wskazane przez najwięcej metod (szereg zmienności) i zdarzenia katalogu

| dzień | liczba metod (z 13) | log RV | zdarzenie w katalogu |
|---|---|---|---|
| 2020-03-18 | 11 | -2,98 | C: luka overnight (COVID); D: COVID - zmienność do 6,4× typowej |
| 2020-03-17 | 11 | -3,02 | D: COVID - zmienność do 6,4× typowej |
| 2020-03-16 | 11 | -2,94 | C: luka overnight −11,6% (COVID); D: COVID - zmienność do 6,4× typowej |
| 2020-03-13 | 11 | -3,01 | D: COVID - zmienność do 6,4× typowej |
| 2020-03-12 | 11 | -2,79 | A: COVID - dzień ze wstrzymaniem handlu; D: COVID - zmienność do 6,4× typowej |
| 2020-03-11 | 11 | -3,78 | D: COVID - zmienność do 6,4× typowej |
| 2020-03-10 | 11 | -3,44 | D: COVID - zmienność do 6,4× typowej |
| 2025-04-09 | 11 | -3,09 | C: ogłoszenie 90-dniowego zawieszenia ceł - przesunięcie poziomu o ok. +6%; D: cła - zmienność 2× (dzień 2025-04-07: 12×); F: cła - wolumen 3,0–3,8× |
| 2025-04-10 | 11 | -3,37 | D: cła - zmienność 2× (dzień 2025-04-07: 12×) |
| 2025-04-11 | 11 | -3,96 | D: cła - zmienność 2× (dzień 2025-04-07: 12×) |
| 2020-03-09 | 11 | -3,34 | C: luka overnight −7% (COVID); D: COVID - zmienność do 6,4× typowej |
| 2020-03-06 | 11 | -3,77 | D: COVID - zmienność do 6,4× typowej |
| 2020-03-05 | 11 | -3,99 | D: COVID - zmienność do 6,4× typowej |
| 2025-04-08 | 11 | -3,54 | D: cła - zmienność 2× (dzień 2025-04-07: 12×); F: cła - wolumen 3,0–3,8× |
| 2025-04-07 | 11 | -2,74 | A: fałszywa informacja o wstrzymaniu ceł - skok i powrót w 10 min; D: cła - zmienność 2× (dzień 2025-04-07: 12×); F: cła - wolumen 3,0–3,8× |

## BTC – szereg: zmiennosc; recall@N (N = 1 / 2 / 5% dni) według typu zdarzenia; wykrycie przy progu z kalibracji (α = 2%) w okresie oceny i liczba dni z alarmem na rok

| metoda | A punktowe (n = 5) | B kolektywne (n = 3) | C zmiana poziomu (n = 6) | D reżim zmienności (n = 7) | wykryte w okresie oceny (próg α = 2%) | dni z alarmem na rok (ocena) |
|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0% / 0% / 0% | 0% / 0% / 0% | 17% / 17% / 33% | 14% / 14% / 14% | 0% | 0,0 |
| PELT (koszt AR) | 0% / 0% / 0% | 0% / 0% / 0% | 0% / 0% / 0% | 0% / 14% / 14% | 36% | 216,5 |
| Matrix Profile (reszty AR) | 40% / 60% / 100% | 33% / 67% / 100% | 33% / 50% / 50% | 43% / 57% / 71% | 36% | 2,5 |
| cechy okien + kNN (5 cech) | 40% / 60% / 80% | 67% / 67% / 100% | 17% / 17% / 50% | 71% / 71% / 71% | 57% | 4,6 |
| cechy okien + kNN (3 cechy) | 60% / 60% / 80% | 33% / 100% / 100% | 17% / 17% / 33% | 57% / 71% / 86% | 57% | 4,3 |
| Isolation Forest (5 cech) | 60% / 60% / 100% | 33% / 67% / 100% | 17% / 17% / 33% | 57% / 71% / 71% | 57% | 5,2 |
| Isolation Forest (3 cechy) | 40% / 60% / 60% | 33% / 67% / 100% | 33% / 33% / 33% | 57% / 71% / 86% | 50% | 3,7 |
| One-Class SVM (5 cech) | 40% / 60% / 80% | 33% / 33% / 67% | 17% / 17% / 17% | 57% / 57% / 71% | 36% | 4,7 |
| One-Class SVM (3 cechy) | 40% / 40% / 60% | 33% / 67% / 67% | 17% / 17% / 33% | 57% / 71% / 86% | 36% | 4,7 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 40% / 40% / 60% | 33% / 67% / 100% | 17% / 17% / 17% | 57% / 71% / 71% | 36% | 2,1 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 40% / 40% / 60% | 33% / 67% / 67% | 17% / 33% / 33% | 57% / 71% / 86% | 50% | 4,0 |
| One-Class SVM (5 cech, uczony na okresie kalibracyjnym) | 40% / 40% / 40% | 33% / 67% / 67% | 17% / 17% / 33% | 57% / 71% / 86% | 36% | 2,8 |
| One-Class SVM (3 cechy, uczony na okresie kalibracyjnym) | 40% / 40% / 60% | 33% / 67% / 67% | 17% / 33% / 50% | 57% / 71% / 71% | 50% | 3,0 |
| CUSUM z S1 (bez modelu AR) | 0% / 0% / 0% | 0% / 0% / 0% | 17% / 17% / 33% | 14% / 14% / 14% | 0% | 0,0 |

## BTC – szereg: wolumen; recall@N (N = 1 / 2 / 5% dni) według typu zdarzenia; wykrycie przy progu z kalibracji (α = 2%) w okresie oceny i liczba dni z alarmem na rok

| metoda | F wolumen (n = 4) | wykryte w okresie oceny (próg α = 2%) | dni z alarmem na rok (ocena) |
|---|---|---|---|
| reszty AR + CUSUM | 0% / 0% / 0% | 25% | 65,0 |
| PELT (koszt AR) | 0% / 0% / 0% | 50% | 187,6 |
| Matrix Profile (reszty AR) | 25% / 25% / 100% | 50% | 8,9 |
| cechy okien + kNN (5 cech) | 50% / 75% / 75% | 75% | 5,6 |
| cechy okien + kNN (3 cechy) | 25% / 50% / 75% | 25% | 5,2 |
| Isolation Forest (5 cech) | 25% / 25% / 75% | 25% | 4,1 |
| Isolation Forest (3 cechy) | 25% / 75% / 100% | 0% | 1,0 |
| One-Class SVM (5 cech) | 25% / 50% / 75% | 25% | 3,4 |
| One-Class SVM (3 cechy) | 25% / 25% / 100% | 25% | 2,2 |
| Isolation Forest (5 cech, uczony na okresie kalibracyjnym) | 0% / 0% / 50% | 0% | 4,0 |
| Isolation Forest (3 cechy, uczony na okresie kalibracyjnym) | 25% / 25% / 100% | 0% | 1,2 |
| One-Class SVM (5 cech, uczony na okresie kalibracyjnym) | 25% / 25% / 50% | 25% | 4,1 |
| One-Class SVM (3 cechy, uczony na okresie kalibracyjnym) | 25% / 50% / 75% | 25% | 4,1 |
| CUSUM z S1 (bez modelu AR) | 0% / 0% / 0% | 0% | 0,0 |

## BTC – zgodność metod (indeks Jaccarda zbiorów 2% najwyższych dni; bez CUSUM z S1)

| szereg | mediana | min | max |
|---|---|---|---|
| wolumen | 0,16 | 0,00 | 0,68 |
| zmiennosc | 0,31 | 0,00 | 0,63 |

## BTC – dni wskazane przez najwięcej metod (szereg zmienności) i zdarzenia katalogu

| dzień | liczba metod (z 13) | log RV | zdarzenie w katalogu |
|---|---|---|---|
| 2020-03-10 | 11 | -3,20 | – |
| 2018-10-11 | 11 | -3,25 | D: reżim spokoju (0,5×) |
| 2018-10-12 | 11 | -4,26 | D: reżim spokoju (0,5×) |
| 2018-10-13 | 11 | -4,51 | D: reżim spokoju (0,5×) |
| 2020-03-16 | 11 | -1,95 | – |
| 2020-03-15 | 11 | -2,35 | – |
| 2023-08-17 | 11 | -2,29 | A: flash crash w najspokojniejszym okresie (−9,1% w 5 min); D: reżim spokoju (0,5×), tło flash crashu 2023-08-17 |
| 2023-08-16 | 11 | -4,46 | D: reżim spokoju (0,5×), tło flash crashu 2023-08-17 |
| 2023-08-15 | 11 | -4,58 | D: reżim spokoju (0,5×), tło flash crashu 2023-08-17 |
| 2018-10-14 | 11 | -4,46 | D: reżim spokoju (0,5×) |
| 2023-08-14 | 11 | -4,53 | D: reżim spokoju (0,5×), tło flash crashu 2023-08-17 |
| 2023-08-13 | 11 | -5,48 | D: reżim spokoju (0,5×), tło flash crashu 2023-08-17 |
| 2018-10-15 | 11 | -2,23 | B: utrata parytetu USDT - anomalia notowania BTC/USDT; D: reżim spokoju (0,5×) |
| 2017-12-02 | 11 | -2,89 | D: bańka i krach - zmienność 2–3,7× |
| 2020-03-13 | 11 | -1,10 | A: odbicie po krachu COVID (+16,9% w 5 min); D: COVID - dzienna RV 9–13×; F: wolumen 5,2× |

## Studia przypadków 5-min (z – |r| względem typowego o tej porze dnia)

| zbiór | studium | dzień | maks. z | mediana z | godzina maks. | zwrot maks. % | zwrot dnia % | metody w top 2% |
|---|---|---|---|---|---|---|---|---|
| SPY | marzec 2020 (COVID) | 2020-03-09 | 28.6 | 7.0 | 14:50 | 0.86 | -0.28 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-10 | 23.6 | 7.5 | 14:40 | 0.71 | 1.34 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-11 | 21.4 | 5.5 | 15:25 | 0.70 | -2.51 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-12 | 63.5 | 8.1 | 12:50 | 1.92 | -1.40 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-13 | 37.0 | 8.9 | 15:30 | -1.46 | 3.68 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-16 | 44.0 | 12.1 | 14:30 | 1.43 | -1.06 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-17 | 44.5 | 9.9 | 12:05 | 1.42 | 3.84 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-18 | 35.6 | 10.6 | 14:35 | 1.09 | 0.54 | 11 |
| SPY | marzec 2020 (COVID) | 2020-03-19 | 32.0 | 9.5 | 13:50 | 0.91 | 2.32 | 9 |
| SPY | marzec 2020 (COVID) | 2020-03-20 | 34.2 | 9.6 | 11:20 | -1.27 | -5.62 | 7 |
| SPY | 2024-08-05 (carry trade) | 2024-08-02 | 8.5 | 1.9 | 10:05 | -0.42 | -0.20 | 0 |
| SPY | 2024-08-05 (carry trade) | 2024-08-05 | 14.7 | 3.2 | 11:05 | 0.57 | 1.01 | 0 |
| SPY | 2024-08-05 (carry trade) | 2024-08-06 | 12.8 | 2.1 | 14:55 | -0.39 | 0.53 | 0 |
| SPY | 2024-08-05 (carry trade) | 2024-08-07 | 10.7 | 2.5 | 15:00 | 0.41 | -1.91 | 0 |
| SPY | kwiecień 2025 (cła) | 2025-04-03 | 9.9 | 2.1 | 12:30 | -0.31 | -1.80 | 9 |
| SPY | kwiecień 2025 (cła) | 2025-04-04 | 25.1 | 6.4 | 11:25 | -0.95 | -3.50 | 9 |
| SPY | kwiecień 2025 (cła) | 2025-04-07 | 77.9 | 8.4 | 10:10 | 3.81 | 3.48 | 11 |
| SPY | kwiecień 2025 (cła) | 2025-04-08 | 24.5 | 6.2 | 15:50 | 1.33 | -4.56 | 11 |
| SPY | kwiecień 2025 (cła) | 2025-04-09 | 71.7 | 5.5 | 13:20 | 2.07 | 9.70 | 11 |
| SPY | kwiecień 2025 (cła) | 2025-04-10 | 31.2 | 7.6 | 12:30 | 0.99 | -1.37 | 11 |
| SPY | 2025-10-10 (cła, likwidacje) | 2025-10-09 | 3.5 | 0.6 | 12:50 | -0.11 | -0.32 | 0 |
| SPY | 2025-10-10 (cła, likwidacje) | 2025-10-10 | 14.5 | 2.0 | 10:55 | -0.55 | -2.88 | 0 |
| SPY | 2025-10-10 (cła, likwidacje) | 2025-10-13 | 4.1 | 1.2 | 09:55 | 0.19 | 0.35 | 1 |
| BTC | marzec 2020 (COVID) | 2020-03-09 | 21.1 | 2.2 | 04:10 | -1.29 | -2.23 | 10 |
| BTC | marzec 2020 (COVID) | 2020-03-10 | 19.9 | 1.5 | 15:30 | -2.02 | -0.57 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-11 | 16.1 | 1.5 | 22:30 | 1.31 | 0.55 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-12 | 179.6 | 4.4 | 10:55 | -10.79 | -39.70 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-13 | 225.8 | 12.6 | 02:35 | 16.91 | 14.99 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-14 | 33.0 | 4.0 | 10:25 | 2.11 | -6.53 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-15 | 40.5 | 2.9 | 21:55 | 2.77 | 4.48 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-16 | 62.2 | 4.3 | 10:00 | -4.91 | -4.27 | 11 |
| BTC | marzec 2020 (COVID) | 2020-03-17 | 27.9 | 3.7 | 12:55 | -1.91 | 5.58 | 4 |
| BTC | marzec 2020 (COVID) | 2020-03-18 | 37.3 | 3.1 | 13:25 | 2.86 | 0.94 | 0 |
| BTC | marzec 2020 (COVID) | 2020-03-19 | 25.3 | 3.4 | 22:55 | -1.71 | 13.43 | 0 |
| BTC | marzec 2020 (COVID) | 2020-03-20 | 53.0 | 4.0 | 08:15 | 4.00 | -0.43 | 0 |
| BTC | 2024-08-05 (carry trade) | 2024-08-02 | 10.0 | 1.4 | 21:10 | -0.78 | -6.07 | 0 |
| BTC | 2024-08-05 (carry trade) | 2024-08-03 | 9.4 | 1.1 | 19:45 | 0.72 | -1.43 | 0 |
| BTC | 2024-08-05 (carry trade) | 2024-08-04 | 17.2 | 1.1 | 17:40 | -1.31 | -4.22 | 0 |
| BTC | 2024-08-05 (carry trade) | 2024-08-05 | 47.7 | 5.3 | 06:20 | -3.10 | -7.34 | 0 |
| BTC | 2024-08-05 (carry trade) | 2024-08-06 | 15.4 | 1.9 | 13:25 | 1.18 | 3.39 | 0 |
| BTC | 2024-08-05 (carry trade) | 2024-08-07 | 9.6 | 1.9 | 05:40 | -0.58 | -1.41 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-03 | 7.0 | 1.2 | 13:30 | 0.76 | 1.08 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-04 | 11.3 | 1.3 | 10:05 | -0.82 | 0.74 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-05 | 5.0 | 0.7 | 23:50 | 0.34 | -0.28 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-06 | 8.5 | 0.9 | 23:35 | 0.61 | -6.08 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-07 | 23.1 | 2.6 | 14:20 | -2.20 | 1.75 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-08 | 8.8 | 1.4 | 01:35 | 0.69 | -3.72 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-09 | 27.4 | 1.9 | 17:20 | 2.27 | 8.60 | 0 |
| BTC | kwiecień 2025 (cła) | 2025-04-10 | 8.9 | 1.3 | 15:35 | -0.86 | -3.63 | 0 |
| BTC | 2025-10-10 (cła, likwidacje) | 2025-10-09 | 5.0 | 0.8 | 16:55 | -0.39 | -1.27 | 0 |
| BTC | 2025-10-10 (cła, likwidacje) | 2025-10-10 | 96.8 | 1.0 | 21:15 | -7.62 | -7.33 | 0 |
| BTC | 2025-10-10 (cła, likwidacje) | 2025-10-11 | 11.2 | 1.2 | 04:50 | 0.66 | -1.89 | 0 |
| BTC | 2025-10-10 (cła, likwidacje) | 2025-10-12 | 11.1 | 1.1 | 03:05 | 0.78 | 4.05 | 0 |
| BTC | 2025-10-10 (cła, likwidacje) | 2025-10-13 | 8.5 | 0.9 | 17:55 | 0.58 | 0.07 | 1 |

## Typ E – dni FOMC w SPY (z o 14:00–14:55 i w pozostałej części dnia; zwykłe dni o 14: mediana 1)

| dzień | maks. z 14:00–14:55 | mediana z 14:00–14:55 | mediana z reszta dnia | metody w top 2% | dni o wyższym log RV (%) |
|---|---|---|---|---|---|
| 2021-06-16 | 14.3 | 4.8 | 0.7 | 0 | 23% |
| 2022-09-21 | 43.7 | 11.9 | 1.5 | 3 | 1% |
| 2023-12-13 | 13.1 | 2.1 | 0.5 | 0 | 32% |
| 2024-12-18 | 13.5 | 5.4 | 0.8 | 0 | 5% |
| 2025-10-29 | 23.5 | 1.6 | 0.8 | 0 | 20% |
