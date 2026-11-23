# Porównanie S1–S3 – tabele (generowane automatycznie przez src/ewaluacja/wykresy_porownanie.py)

Nazwy metod w rodzinach:

| rodzina | S1 | S2 | S3 |
|---|---|---|---|
| karta kontrolna | CUSUM (poziom, zmienność, zależność) | reszty AR + CUSUM | STL + EWMA |
| PELT | PELT | PELT (koszt AR) | PELT (reszty STL) |
| Matrix Profile | Matrix Profile | Matrix Profile (reszty AR) | Matrix Profile z normalizacją |
| kNN | cechy okien + kNN | cechy okien + kNN | cechy cykli + kNN |
| Isolation Forest | Isolation Forest | Isolation Forest | Isolation Forest |
| druga metoda ML | LOF | One-Class SVM | autoenkoder |

## Odsetek wykrytych – część główna (wszystkie poziomy i L)

| metoda | S1-A poziom | S1-B zmienność | S1-C zależność | S2-A szok | S2-B zmiana φ | S2-C równowaga | S3-A faza | S3-B amplituda | S3-C okres |
|---|---|---|---|---|---|---|---|---|---|
| karta kontrolna | 76% | 84% | 39% | 31% | 66% | 45% | 82% | 67% | 93% |
| PELT | 66% | 88% | 19% | 23% | 55% | 36% | 84% | 72% | 93% |
| Matrix Profile | 64% | 91% | 6% | 31% | 66% | 24% | 61% | 87% | 90% |
| kNN – 5 cech | 57% | 76% | 11% | 42% | 60% | 37% | 91% | 86% | 96% |
| kNN – 3 cechy | 77% | 87% | 42% | 42% | 48% | 38% | 92% | 87% | 96% |
| Isolation Forest – 5 cech | 26% | 75% | 2% | 8% | 48% | 1% | 31% | 25% | 78% |
| Isolation Forest – 3 cechy | 40% | 45% | 3% | 14% | 18% | 6% | 26% | 39% | 77% |
| Isolation Forest – 5 cech, uczony bez anomalii | 53% | 77% | 19% | 11% | 54% | 16% | 76% | 82% | 94% |
| Isolation Forest – 3 cechy, uczony bez anomalii | 59% | 56% | 32% | 15% | 38% | 18% | 74% | 83% | 93% |
| druga metoda ML – 5 cech | 63% | 72% | 26% | 4% | 17% | 0% | 68% | 62% | 93% |
| druga metoda ML – 3 cechy | 68% | 78% | 36% | 8% | 14% | 5% | – | – | – |
| druga metoda ML – 5 cech, uczony bez anomalii | 77% | 82% | 36% | 41% | 60% | 38% | 67% | 46% | 92% |
| druga metoda ML – 3 cechy, uczony bez anomalii | 79% | 84% | 35% | 42% | 48% | 39% | – | – | – |

## AUC – część główna (wszystkie poziomy i L)

| metoda | S1-A poziom | S1-B zmienność | S1-C zależność | S2-A szok | S2-B zmiana φ | S2-C równowaga | S3-A faza | S3-B amplituda | S3-C okres |
|---|---|---|---|---|---|---|---|---|---|
| karta kontrolna | 0,90 | 0,92 | 0,88 | 0,61 | 0,79 | 0,82 | 0,95 | 0,89 | 0,93 |
| PELT | 0,81 | 0,90 | 0,55 | 0,56 | 0,75 | 0,66 | 0,86 | 0,79 | 0,85 |
| Matrix Profile | 0,90 | 0,99 | 0,46 | 0,80 | 0,92 | 0,75 | 0,68 | 0,96 | 0,88 |
| kNN – 5 cech | 0,94 | 0,97 | 0,83 | 0,81 | 0,91 | 0,82 | 0,99 | 0,98 | 0,99 |
| kNN – 3 cechy | 0,95 | 0,98 | 0,85 | 0,81 | 0,87 | 0,83 | 0,99 | 0,98 | 0,99 |
| Isolation Forest – 5 cech | 0,94 | 0,98 | 0,84 | 0,80 | 0,90 | 0,82 | 0,99 | 0,98 | 0,99 |
| Isolation Forest – 3 cechy | 0,95 | 0,98 | 0,86 | 0,81 | 0,86 | 0,84 | 0,99 | 0,98 | 0,99 |
| Isolation Forest – 5 cech, uczony bez anomalii | 0,93 | 0,98 | 0,83 | 0,80 | 0,90 | 0,82 | 0,99 | 0,98 | 0,99 |
| Isolation Forest – 3 cechy, uczony bez anomalii | 0,96 | 0,97 | 0,86 | 0,80 | 0,86 | 0,84 | 0,99 | 0,98 | 0,99 |
| druga metoda ML – 5 cech | 0,92 | 0,91 | 0,81 | 0,75 | 0,81 | 0,80 | 0,71 | 0,72 | 0,93 |
| druga metoda ML – 3 cechy | 0,92 | 0,93 | 0,82 | 0,74 | 0,71 | 0,80 | – | – | – |
| druga metoda ML – 5 cech, uczony bez anomalii | 0,95 | 0,96 | 0,82 | 0,78 | 0,88 | 0,85 | 0,71 | 0,67 | 0,93 |
| druga metoda ML – 3 cechy, uczony bez anomalii | 0,95 | 0,97 | 0,83 | 0,77 | 0,80 | 0,85 | – | – | – |

## Ranking metod w scenariuszu (średnio A, B, C; wszystkie poziomy i L)

| scenariusz | miejsce | metoda | wykryte (śr.) | AUC (śr.) |
|---|---|---|---|---|
| S1 | 1 | cechy okien + kNN (3 cechy) | 69% | 0,93 |
| S1 | 2 | CUSUM (poziom, zmienność, zależność) | 66% | 0,90 |
| S1 | 3 | LOF (3 cechy, uczony na szeregach bez anomalii) | 66% | 0,92 |
| S1 | 4 | LOF (5 cech, uczony na szeregach bez anomalii) | 65% | 0,91 |
| S1 | 5 | LOF (3 cechy) | 61% | 0,89 |
| S1 | 6 | PELT | 58% | 0,76 |
| S1 | 7 | LOF (5 cech) | 54% | 0,88 |
| S1 | 8 | Matrix Profile | 53% | 0,78 |
| S1 | 9 | Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 50% | 0,91 |
| S1 | 10 | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 49% | 0,93 |
| S1 | 11 | cechy okien + kNN (5 cech) | 48% | 0,92 |
| S1 | 12 | Isolation Forest (5 cech) | 34% | 0,92 |
| S1 | 13 | Isolation Forest (3 cechy) | 29% | 0,93 |
| S2 | 1 | reszty AR + CUSUM | 47% | 0,74 |
| S2 | 2 | cechy okien + kNN (5 cech) | 46% | 0,85 |
| S2 | 3 | One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 46% | 0,84 |
| S2 | 4 | One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 43% | 0,81 |
| S2 | 5 | cechy okien + kNN (3 cechy) | 43% | 0,84 |
| S2 | 6 | Matrix Profile (reszty AR) | 40% | 0,82 |
| S2 | 7 | PELT (koszt AR) | 38% | 0,66 |
| S2 | 8 | Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 27% | 0,84 |
| S2 | 9 | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 23% | 0,84 |
| S2 | 10 | Isolation Forest (5 cech) | 19% | 0,84 |
| S2 | 11 | Isolation Forest (3 cechy) | 13% | 0,83 |
| S2 | 12 | One-Class SVM (3 cechy) | 9% | 0,75 |
| S2 | 13 | One-Class SVM (5 cech) | 7% | 0,79 |
| S3 | 1 | cechy cykli + kNN (3 cechy) | 91% | 0,99 |
| S3 | 2 | cechy cykli + kNN (5 cech) | 91% | 0,99 |
| S3 | 3 | Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 84% | 0,99 |
| S3 | 4 | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 83% | 0,99 |
| S3 | 5 | PELT (reszty STL) | 83% | 0,83 |
| S3 | 6 | STL + EWMA | 80% | 0,92 |
| S3 | 7 | Matrix Profile z normalizacją | 80% | 0,84 |
| S3 | 8 | autoenkoder | 74% | 0,79 |
| S3 | 9 | autoenkoder (uczony na szeregach bez anomalii) | 69% | 0,77 |
| S3 | 10 | Isolation Forest (3 cechy) | 47% | 0,99 |
| S3 | 11 | Isolation Forest (5 cech) | 45% | 0,98 |

## Najlepsza metoda dla każdej anomalii – w każdej grupie metod (wszystkie poziomy i L)

| kategoria | anomalia | statystyczne | fragmenty | ML |
|---|---|---|---|---|
| wartości | S1-A poziom | CUSUM (poziom, zmienność, zależność) – 76% | cechy okien + kNN (3 cechy) – 77% | LOF (3 cechy, uczony na szeregach bez anomalii) – 79% |
| wartości | S1-B zmienność | PELT – 88% | Matrix Profile – 91% | LOF (3 cechy, uczony na szeregach bez anomalii) – 84% |
| wartości | S3-B amplituda | PELT (reszty STL) – 72% | Matrix Profile z normalizacją – 87% | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) – 83% |
| struktury | S1-C zależność | CUSUM (poziom, zmienność, zależność) – 39% | cechy okien + kNN (3 cechy) – 42% | LOF (5 cech, uczony na szeregach bez anomalii) – 36% |
| struktury | S2-B zmiana φ | reszty AR + CUSUM – 66% | Matrix Profile (reszty AR) – 66% | One-Class SVM (5 cech, uczony na szeregach bez anomalii) – 60% |
| struktury | S3-A faza | PELT (reszty STL) – 84% | cechy cykli + kNN (3 cechy) – 92% | Isolation Forest (5 cech, uczony na szeregach bez anomalii) – 76% |
| struktury | S3-C okres | PELT (reszty STL) – 93% | cechy cykli + kNN (5 cech) – 96% | Isolation Forest (5 cech, uczony na szeregach bez anomalii) – 94% |
| dynamiki | S2-A szok | reszty AR + CUSUM – 31% | cechy okien + kNN (3 cechy) – 42% | One-Class SVM (3 cechy, uczony na szeregach bez anomalii) – 42% |
| dynamiki | S2-C równowaga | reszty AR + CUSUM – 45% | cechy okien + kNN (3 cechy) – 38% | One-Class SVM (3 cechy, uczony na szeregach bez anomalii) – 39% |

Średnio w kategorii (dla każdej rodziny najlepszy wariant):

| kategoria | karta kontrolna | PELT | Matrix Profile | kNN | Isolation Forest | druga metoda ML |
|---|---|---|---|---|---|---|
| wartości | 76% | 75% | 81% | 84% | 73% | 75% |
| struktury | 70% | 63% | 56% | 72% | 64% | 64% |
| dynamiki | 38% | 30% | 27% | 40% | 16% | 41% |

## Przeniesienie metod – odsetek wykrytych A / B / C (poziomy 2–3, wszystkie L)

Szeregi S1 – biały szum:

| metoda | metody S1 (u siebie) | metody S2 | metody S3 |
|---|---|---|---|
| karta kontrolna | 92 / 95 / 38 | 88 / 95 / 32 | 12 / 96 / 1 |
| PELT | 79 / 97 / 13 | 86 / 93 / 39 | 17 / 94 / 8 |
| Matrix Profile | 74 / 97 / 4 | 71 / 97 / 2 | 0 / 0 / 0 |
| kNN – 5 cech | 64 / 90 / 1 | 87 / 95 / 35 | 69 / 94 / 19 |
| kNN – 3 cechy | 89 / 96 / 43 | 87 / 87 / 29 | 59 / 35 / 4 |
| Isolation Forest – 5 cech | 17 / 87 / 0 | 46 / 89 / 8 | 33 / 34 / 3 |
| Isolation Forest – 3 cechy | 39 / 47 / 0 | 55 / 56 / 2 | 48 / 14 / 1 |
| druga metoda ML – 5 cech | 75 / 81 / 18 | 20 / 38 / 0 | 71 / 97 / 3 |
| druga metoda ML – 3 cechy | 80 / 88 / 31 | 45 / 19 / 1 | – |

Szeregi S2 – AR(1), φ = 0,7:

| metoda | metody S1 | metody S2 (u siebie) | metody S3 |
|---|---|---|---|
| karta kontrolna | 34 / 51 / 49 | 13 / 73 / 50 | 4 / 0 / 6 |
| PELT | 5 / 7 / 40 | 3 / 61 / 33 | 2 / 2 / 4 |
| Matrix Profile | 1 / 0 / 41 | 15 / 74 / 13 | 0 / 37 / 0 |
| kNN – 5 cech | 12 / 10 / 14 | 34 / 65 / 32 | 0 / 0 / 12 |
| kNN – 3 cechy | 0 / 36 / 42 | 35 / 50 / 36 | 0 / 0 / 14 |
| Isolation Forest – 5 cech | 3 / 0 / 15 | 3 / 55 / 1 | 0 / 0 / 12 |
| Isolation Forest – 3 cechy | 0 / 0 / 11 | 7 / 19 / 4 | 0 / 0 / 13 |
| druga metoda ML – 5 cech | 8 / 19 / 27 | 4 / 24 / 0 | 1 / 1 / 18 |
| druga metoda ML – 3 cechy | 0 / 24 / 33 | 8 / 18 / 4 | – |

Szeregi S3 – szum + sinus (A = 3, P = 50):

| metoda | metody S1 | metody S2 | metody S3 (u siebie) |
|---|---|---|---|
| karta kontrolna | 4 / 0 / 14 | 1 / 1 / 3 | 99 / 83 / 100 |
| PELT | 1 / 93 / 0 | 0 / 23 / 1 | 100 / 94 / 100 |
| Matrix Profile | 79 / 86 / 100 | 3 / 1 / 8 | 72 / 100 / 100 |
| kNN – 5 cech | 52 / 98 / 75 | 58 / 98 / 88 | 100 / 98 / 100 |
| kNN – 3 cechy | 66 / 98 / 92 | 62 / 92 / 90 | 100 / 99 / 100 |
| Isolation Forest – 5 cech | 21 / 96 / 36 | 5 / 39 / 17 | 20 / 14 / 100 |
| Isolation Forest – 3 cechy | 35 / 83 / 51 | 18 / 6 / 44 | 19 / 28 / 97 |
| druga metoda ML – 5 cech | 37 / 77 / 41 | 13 / 42 / 42 | 81 / 73 / 100 |
| druga metoda ML – 3 cechy | 33 / 74 / 45 | 39 / 61 / 66 | – |

## Przeniesienie metod – AUC A / B / C (poziomy 2–3, wszystkie L)

Szeregi S1 – biały szum:

| metoda | metody S1 (u siebie) | metody S2 | metody S3 |
|---|---|---|---|
| karta kontrolna | 0,98 / 0,96 / 0,92 | 0,99 / 0,97 / 0,91 | 0,71 / 0,97 / 0,37 |
| PELT | 0,87 / 0,95 / 0,52 | 0,91 / 0,94 / 0,67 | 0,56 / 0,94 / 0,51 |
| Matrix Profile | 0,95 / 1,00 / 0,47 | 0,95 / 1,00 / 0,44 | 0,47 / 0,50 / 0,45 |
| kNN – 5 cech | 0,99 / 1,00 / 0,85 | 0,99 / 1,00 / 0,85 | 0,83 / 1,00 / 0,72 |
| kNN – 3 cechy | 0,99 / 1,00 / 0,88 | 0,99 / 0,98 / 0,86 | 0,82 / 0,80 / 0,69 |
| Isolation Forest – 5 cech | 0,98 / 1,00 / 0,87 | 0,99 / 1,00 / 0,87 | 0,83 / 0,99 / 0,70 |
| Isolation Forest – 3 cechy | 0,99 / 1,00 / 0,88 | 0,99 / 0,98 / 0,87 | 0,82 / 0,77 / 0,64 |
| druga metoda ML – 5 cech | 0,97 / 0,95 / 0,84 | 0,93 / 0,97 / 0,77 | 0,88 / 0,98 / 0,46 |
| druga metoda ML – 3 cechy | 0,97 / 0,97 / 0,84 | 0,94 / 0,89 / 0,78 | – |

Szeregi S2 – AR(1), φ = 0,7:

| metoda | metody S1 | metody S2 (u siebie) | metody S3 |
|---|---|---|---|
| karta kontrolna | 0,54 / 0,53 / 0,82 | 0,61 / 0,86 / 0,87 | 0,67 / 0,64 / 0,63 |
| PELT | 0,49 / 0,51 / 0,67 | 0,50 / 0,77 / 0,64 | 0,50 / 0,50 / 0,51 |
| Matrix Profile | 0,74 / 0,77 / 0,85 | 0,81 / 0,95 / 0,74 | 0,49 / 0,86 / 0,34 |
| kNN – 5 cech | 0,74 / 0,90 / 0,88 | 0,82 / 0,95 / 0,86 | 0,62 / 0,55 / 0,65 |
| kNN – 3 cechy | 0,67 / 0,92 / 0,89 | 0,83 / 0,91 / 0,88 | 0,53 / 0,54 / 0,62 |
| Isolation Forest – 5 cech | 0,72 / 0,83 / 0,86 | 0,81 / 0,95 / 0,86 | 0,60 / 0,51 / 0,65 |
| Isolation Forest – 3 cechy | 0,64 / 0,89 / 0,90 | 0,82 / 0,91 / 0,89 | 0,52 / 0,55 / 0,63 |
| druga metoda ML – 5 cech | 0,68 / 0,90 / 0,86 | 0,73 / 0,86 / 0,84 | 0,74 / 0,68 / 0,67 |
| druga metoda ML – 3 cechy | 0,64 / 0,90 / 0,86 | 0,73 / 0,76 / 0,84 | – |

Szeregi S3 – szum + sinus (A = 3, P = 50):

| metoda | metody S1 | metody S2 | metody S3 (u siebie) |
|---|---|---|---|
| karta kontrolna | 0,52 / 0,50 / 0,51 | 0,53 / 0,27 / 0,47 | 0,98 / 0,94 / 0,96 |
| PELT | 0,50 / 0,95 / 0,50 | 0,50 / 0,61 / 0,50 | 0,95 / 0,86 / 0,90 |
| Matrix Profile | 0,73 / 0,92 / 0,95 | 0,58 / 0,52 / 0,70 | 0,71 / 1,00 / 0,95 |
| kNN – 5 cech | 0,69 / 1,00 / 0,89 | 0,70 / 1,00 / 0,90 | 1,00 / 1,00 / 1,00 |
| kNN – 3 cechy | 0,70 / 1,00 / 0,91 | 0,70 / 0,99 / 0,90 | 1,00 / 1,00 / 1,00 |
| Isolation Forest – 5 cech | 0,69 / 1,00 / 0,84 | 0,68 / 0,99 / 0,84 | 1,00 / 1,00 / 1,00 |
| Isolation Forest – 3 cechy | 0,70 / 1,00 / 0,89 | 0,69 / 0,98 / 0,88 | 1,00 / 1,00 / 1,00 |
| druga metoda ML – 5 cech | 0,69 / 0,84 / 0,87 | 0,63 / 0,90 / 0,73 | 0,75 / 0,76 / 1,00 |
| druga metoda ML – 3 cechy | 0,69 / 0,84 / 0,87 | 0,63 / 0,85 / 0,77 | – |

Średnio dla wszystkich wariantów metod (A, B, C łącznie) i najlepszy wariant:

| szeregi | metody S1 | metody S2 | metody S3 |
|---|---|---|---|
| S1 – biały szum | 57% (najlepsza: kNN – 3 cechy – 76%) | 52% (najlepsza: PELT – 73%) | 34% (najlepsza: kNN – 5 cech – 61%) |
| S2 – AR(1), φ = 0,7 | 18% (najlepsza: karta kontrolna – 45%) | 27% (najlepsza: karta kontrolna – 45%) | 5% (najlepsza: Matrix Profile – 12%) |
| S3 – szum + sinus (A = 3, P = 50) | 55% (najlepsza: Matrix Profile – 89%) | 34% (najlepsza: kNN – 3 cechy – 82%) | 82% (najlepsza: kNN – 3 cechy – 100%) |

Fałszywe alarmy na walidacji (200 szeregów bez anomalii) – metody przeniesione:

| metody z | szeregi | metoda | szeregi z ≥ 1 fałszywym alarmem |
|---|---|---|---|
| S1 | S2 | CUSUM (poziom, zmienność, zależność) | 7,0% |
| S1 | S2 | Isolation Forest (3 cechy) | 6,0% |
| S1 | S2 | Isolation Forest (5 cech) | 7,5% |
| S1 | S2 | LOF (3 cechy) | 10,5% |
| S1 | S2 | LOF (5 cech) | 7,5% |
| S1 | S2 | Matrix Profile | 9,5% |
| S1 | S2 | PELT | 5,5% |
| S1 | S2 | cechy okien + kNN (3 cechy) | 9,0% |
| S1 | S2 | cechy okien + kNN (5 cech) | 7,0% |
| S1 | S3 | CUSUM (poziom, zmienność, zależność) | 13,0% |
| S1 | S3 | Isolation Forest (3 cechy) | 9,0% |
| S1 | S3 | Isolation Forest (5 cech) | 13,5% |
| S1 | S3 | LOF (3 cechy) | 11,0% |
| S1 | S3 | LOF (5 cech) | 13,5% |
| S1 | S3 | Matrix Profile | 13,5% |
| S1 | S3 | PELT | 9,5% |
| S1 | S3 | cechy okien + kNN (3 cechy) | 14,0% |
| S1 | S3 | cechy okien + kNN (5 cech) | 7,5% |
| S2 | S1 | Isolation Forest (3 cechy) | 17,0% |
| S2 | S1 | Isolation Forest (5 cech) | 8,0% |
| S2 | S1 | Matrix Profile (reszty AR) | 12,5% |
| S2 | S1 | One-Class SVM (3 cechy) | 11,0% |
| S2 | S1 | One-Class SVM (5 cech) | 7,5% |
| S2 | S1 | PELT (koszt AR) | 6,5% |
| S2 | S1 | cechy okien + kNN (3 cechy) | 11,0% |
| S2 | S1 | cechy okien + kNN (5 cech) | 12,0% |
| S2 | S1 | reszty AR + CUSUM | 11,0% |
| S2 | S3 | Isolation Forest (3 cechy) | 7,0% |
| S2 | S3 | Isolation Forest (5 cech) | 6,5% |
| S2 | S3 | Matrix Profile (reszty AR) | 18,5% |
| S2 | S3 | One-Class SVM (3 cechy) | 14,5% |
| S2 | S3 | One-Class SVM (5 cech) | 11,5% |
| S2 | S3 | PELT (koszt AR) | 8,0% |
| S2 | S3 | cechy okien + kNN (3 cechy) | 13,0% |
| S2 | S3 | cechy okien + kNN (5 cech) | 9,5% |
| S2 | S3 | reszty AR + CUSUM | 13,5% |
| S3 | S1 | Isolation Forest (3 cechy) | 4,5% |
| S3 | S1 | Isolation Forest (5 cech) | 7,5% |
| S3 | S1 | Matrix Profile z normalizacją | 8,5% |
| S3 | S1 | PELT (reszty STL) | 8,0% |
| S3 | S1 | STL + EWMA | 11,5% |
| S3 | S1 | autoenkoder | 12,5% |
| S3 | S1 | cechy cykli + kNN (3 cechy) | 9,5% |
| S3 | S1 | cechy cykli + kNN (5 cech) | 12,5% |
| S3 | S2 | Isolation Forest (3 cechy) | 8,0% |
| S3 | S2 | Isolation Forest (5 cech) | 9,5% |
| S3 | S2 | Matrix Profile z normalizacją | 13,5% |
| S3 | S2 | PELT (reszty STL) | 9,0% |
| S3 | S2 | STL + EWMA | 14,0% |
| S3 | S2 | autoenkoder | 13,0% |
| S3 | S2 | cechy cykli + kNN (3 cechy) | 7,0% |
| S3 | S2 | cechy cykli + kNN (5 cech) | 8,0% |

## Zestaw cech: 5 cech → 3 cechy (odsetek wykrytych A / B / C)

| metoda | S1 | S2 | S3 |
|---|---|---|---|
| kNN | 57 / 76 / 11 → 77 / 87 / 42 | 42 / 60 / 37 → 42 / 48 / 38 | 91 / 86 / 96 → 92 / 87 / 96 |
| Isolation Forest | 26 / 75 / 2 → 40 / 45 / 3 | 8 / 48 / 1 → 14 / 18 / 6 | 31 / 25 / 78 → 26 / 39 / 77 |
| druga metoda ML | 63 / 72 / 26 → 68 / 78 / 36 | 4 / 17 / 0 → 8 / 14 / 5 | – |

## Uczenie: ten sam szereg → szeregi bez anomalii (odsetek wykrytych A / B / C)

| metoda | S1 | S2 | S3 |
|---|---|---|---|
| Isolation Forest – 3 cechy | 40 / 45 / 3 → 59 / 56 / 32 | 14 / 18 / 6 → 15 / 38 / 18 | 26 / 39 / 77 → 74 / 83 / 93 |
| Isolation Forest – 5 cech | 26 / 75 / 2 → 53 / 77 / 19 | 8 / 48 / 1 → 11 / 54 / 16 | 31 / 25 / 78 → 76 / 82 / 94 |
| druga metoda ML – 3 cechy | 68 / 78 / 36 → 79 / 84 / 35 | 8 / 14 / 5 → 42 / 48 / 39 | – |
| druga metoda ML – 5 cech | 63 / 72 / 26 → 77 / 82 / 36 | 4 / 17 / 0 → 41 / 60 / 38 | 68 / 62 / 93 → 67 / 46 / 92 |

## Właściwość szeregu S1: poziom szumu σ (Δ = 1, c = 2, φ = 0,5 bezwzględnie) – odsetek wykrytych A / B / C

| metoda | sigma = 0,5 | sigma = 1 | sigma = 2 |
|---|---|---|---|
| karta kontrolna | 100 / 99 / 17 | 98 / 99 / 17 | 20 / 99 / 17 |
| PELT | 100 / 100 / 5 | 57 / 100 / 5 | 6 / 100 / 5 |
| Matrix Profile | 100 / 100 / 2 | 64 / 100 / 2 | 11 / 100 / 2 |
| kNN – 5 cech | 100 / 98 / 0 | 42 / 98 / 0 | 0 / 98 / 0 |
| kNN – 3 cechy | 100 / 100 / 31 | 98 / 100 / 33 | 24 / 100 / 32 |
| Isolation Forest – 5 cech | 68 / 80 / 1 | 4 / 88 / 1 | 0 / 94 / 2 |
| Isolation Forest – 3 cechy | 96 / 28 / 0 | 9 / 43 / 3 | 0 / 54 / 9 |
| druga metoda ML – 5 cech | 100 / 91 / 10 | 84 / 99 / 9 | 7 / 100 / 10 |
| druga metoda ML – 3 cechy | 100 / 99 / 27 | 94 / 100 / 28 | 17 / 100 / 27 |

## Właściwość szeregu S2: siła zależności tła φ (szok 5σ_ε, φ' = 0, Δ = 2σ_x) – odsetek wykrytych A / B / C

| metoda | phi_tla = 0,3 | phi_tla = 0,7 | phi_tla = 0,9 |
|---|---|---|---|
| karta kontrolna | 33 / 12 / 100 | 29 / 97 / 90 | 24 / 100 / 23 |
| PELT | 12 / 3 / 100 | 8 / 92 / 59 | 7 / 100 / 3 |
| Matrix Profile | 29 / 5 / 99 | 26 / 95 / 33 | 29 / 100 / 5 |
| kNN – 5 cech | 68 / 5 / 100 | 63 / 97 / 73 | 39 / 100 / 3 |
| kNN – 3 cechy | 68 / 4 / 100 | 62 / 86 / 81 | 38 / 100 / 4 |
| Isolation Forest – 5 cech | 14 / 1 / 98 | 4 / 88 / 1 | 0 / 100 / 0 |
| Isolation Forest – 3 cechy | 8 / 1 / 92 | 9 / 35 / 16 | 1 / 100 / 1 |
| druga metoda ML – 5 cech | 1 / 0 / 96 | 1 / 62 / 2 | 0 / 8 / 0 |
| druga metoda ML – 3 cechy | 4 / 0 / 100 | 2 / 41 / 22 | 0 / 26 / 0 |

## Właściwość szeregu S3: amplituda cyklu A przy σ = 1 (faza π/2, ×0,25, okres ×0,6) – odsetek wykrytych A / B / C

| metoda | A = 1 | A = 3 | A = 5 |
|---|---|---|---|
| karta kontrolna | 63 / 3 / 92 | 100 / 100 / 100 | 100 / 100 / 100 |
| PELT | 84 / 1 / 96 | 100 / 100 / 100 | 100 / 100 / 100 |
| Matrix Profile | 25 / 91 / 53 | 100 / 100 / 100 | 100 / 100 / 100 |
| kNN – 5 cech | 0 / 11 / 3 | 100 / 100 / 100 | 100 / 100 / 100 |
| kNN – 3 cechy | 0 / 7 / 0 | 100 / 100 / 100 | 100 / 100 / 100 |
| Isolation Forest – 5 cech | 0 / 2 / 1 | 8 / 19 / 100 | 34 / 2 / 100 |
| Isolation Forest – 3 cechy | 0 / 6 / 0 | 10 / 35 / 100 | 20 / 16 / 100 |
| druga metoda ML – 5 cech | 13 / 2 / 45 | 100 / 92 / 100 | 100 / 100 / 100 |

## Długość okna w – odsetek wykrytych (średnio A, B, C)

| metoda | S1: w = 20 / 50 / 100 | S2: w = 20 / 50 / 100 | S3: w = 25 / 50 / 100 |
|---|---|---|---|
| karta kontrolna | – | – | – |
| PELT | – | – | – |
| Matrix Profile | 52 / 58 / 58 | 31 / 32 / 29 | 53 / 91 / 99 |
| kNN – 5 cech | 37 / 52 / 61 | 33 / 44 / 40 | 17 / 100 / 100 |
| kNN – 3 cechy | 70 / 76 / 70 | 24 / 38 / 38 | 1 / 99 / 100 |
| Isolation Forest – 5 cech | 34 / 35 / 31 | 21 / 21 / 18 | 18 / 44 / 44 |
| Isolation Forest – 3 cechy | 40 / 29 / 25 | 7 / 9 / 5 | 23 / 46 / 37 |
| druga metoda ML – 5 cech | 42 / 59 / 62 | 22 / 10 / 3 | 54 / 85 / 94 |
| druga metoda ML – 3 cechy | 50 / 68 / 59 | 16 / 9 / 6 | – |

## Długość anomalii L – odsetek wykrytych (średnio A, B, C)

| metoda | S1: L = 20 / 50 / 100 | S2: L = 20 / 50 / 100 | S3: L = 50 / 100 / 200 |
|---|---|---|---|
| karta kontrolna | 48 / 67 / 84 | 31 / 51 / 60 | 72 / 84 / 85 |
| PELT | 44 / 57 / 71 | 22 / 41 / 51 | 75 / 86 / 88 |
| Matrix Profile | 45 / 56 / 60 | 27 / 45 / 49 | 73 / 81 / 85 |
| kNN – 5 cech | 36 / 52 / 56 | 31 / 52 / 56 | 87 / 95 / 90 |
| kNN – 3 cechy | 47 / 74 / 84 | 29 / 48 / 52 | 88 / 96 / 90 |
| Isolation Forest – 5 cech | 35 / 38 / 29 | 14 / 23 / 21 | 60 / 44 / 31 |
| Isolation Forest – 3 cechy | 28 / 40 / 19 | 14 / 17 / 7 | 62 / 45 / 34 |
| druga metoda ML – 5 cech | 38 / 63 / 61 | 5 / 12 / 4 | 70 / 78 / 75 |
| druga metoda ML – 3 cechy | 41 / 70 / 70 | 10 / 13 / 4 | – |

## Fałszywe alarmy (walidacja 200 / sprawdzenie na 1 000 szeregów) i F1

| metoda | S1 | S2 | S3 |
|---|---|---|---|
| karta kontrolna | 10,0% / 9,8% / 0,65 | 10,5% / 8,6% / 0,52 | 13,5% / 12,1% / 0,65 |
| PELT | 9,5% / 8,6% / 0,58 | 7,0% / 4,0% / 0,41 | 6,5% / 7,2% / 0,73 |
| Matrix Profile | 13,0% / 12,4% / 0,56 | 12,0% / 12,0% / 0,36 | 13,5% / 10,4% / 0,55 |
| kNN – 5 cech | 8,5% / 9,7% / 0,48 | 11,5% / 9,3% / 0,42 | 15,5% / 13,1% / 0,80 |
| kNN – 3 cechy | 12,5% / 11,1% / 0,65 | 10,5% / 10,8% / 0,37 | 13,0% / 11,6% / 0,79 |
| Isolation Forest – 5 cech | 10,5% / 9,6% / 0,27 | 10,5% / 10,0% / 0,14 | 13,0% / 10,5% / 0,27 |
| Isolation Forest – 3 cechy | 4,0% / 5,7% / 0,19 | 9,0% / 9,4% / 0,04 | 12,0% / 9,7% / 0,21 |
| Isolation Forest – 5 cech, uczony bez anomalii | 10,5% / 11,1% / 0,36 | 9,5% / 9,9% / 0,26 | 11,0% / 9,3% / 0,61 |
| Isolation Forest – 3 cechy, uczony bez anomalii | 7,5% / 7,8% / 0,33 | 6,5% / 7,8% / 0,20 | 13,0% / 9,7% / 0,55 |
| druga metoda ML – 5 cech | 10,0% / 11,6% / 0,40 | 12,5% / 11,8% / 0,04 | 16,5% / 10,7% / 0,50 |
| druga metoda ML – 3 cechy | 16,5% / 13,2% / 0,44 | 11,0% / 12,3% / 0,05 | – |
| druga metoda ML – 5 cech, uczony bez anomalii | 14,0% / 15,7% / 0,63 | 13,5% / 12,2% / 0,44 | 13,0% / 9,2% / 0,48 |
| druga metoda ML – 3 cechy, uczony bez anomalii | 10,5% / 13,7% / 0,64 | 9,5% / 12,2% / 0,39 | – |
