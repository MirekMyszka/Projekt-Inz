# S2 – tabele wyników (generowane automatycznie przez src/ewaluacja/wykresy_s2.py)

Część główna: 1200 szeregów z anomaliami (3600 anomalii na metodę), 200 szeregów walidacyjnych bez anomalii. Próg: fałszywy alarm w 10% szeregów kalibracyjnych bez anomalii.

## Kalibracja

| metoda                                                       |     prog |   kara_pelt |
|:-------------------------------------------------------------|---------:|------------:|
| reszty AR + CUSUM                                            | 1.61352  |         nan |
| PELT (koszt AR)                                              | 0        |          18 |
| Matrix Profile (reszty AR)                                   | 8.88608  |         nan |
| cechy okien + kNN (5 cech)                                   | 3.53432  |         nan |
| cechy okien + kNN (3 cechy)                                  | 2.95031  |         nan |
| Isolation Forest (5 cech)                                    | 0.676459 |         nan |
| Isolation Forest (3 cechy)                                   | 0.693707 |         nan |
| One-Class SVM (5 cech)                                       | 2.75847  |         nan |
| One-Class SVM (3 cechy)                                      | 2.41747  |         nan |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii)  | 0.663301 |         nan |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0.686837 |         nan |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii)     | 7.28271  |         nan |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii)    | 6.8664   |         nan |

## Fałszywe alarmy (walidacja na nowych szeregach bez anomalii) i miary punktowe

| metoda                                                       |   odsetek_z_fa |   fa_na_1000_pkt |   fa_w_szeregach_z_anomaliami |   precyzja |   czulosc_pkt |    f1 |
|:-------------------------------------------------------------|---------------:|-----------------:|------------------------------:|-----------:|--------------:|------:|
| reszty AR + CUSUM                                            |          0.105 |            0.038 |                         0.05  |      0.932 |         0.468 | 0.516 |
| PELT (koszt AR)                                              |          0.07  |            0.023 |                         0.039 |      0.832 |         0.402 | 0.406 |
| Matrix Profile (reszty AR)                                   |          0.12  |            0.087 |                         0.175 |      0.923 |         0.27  | 0.356 |
| cechy okien + kNN (5 cech)                                   |          0.115 |            0.047 |                         0.061 |      0.95  |         0.353 | 0.419 |
| cechy okien + kNN (3 cechy)                                  |          0.105 |            0.04  |                         0.058 |      0.937 |         0.308 | 0.368 |
| Isolation Forest (5 cech)                                    |          0.105 |            0.037 |                         0.028 |      0.947 |         0.085 | 0.136 |
| Isolation Forest (3 cechy)                                   |          0.09  |            0.03  |                         0.025 |      0.903 |         0.026 | 0.043 |
| One-Class SVM (5 cech)                                       |          0.125 |            0.045 |                         0.033 |      0.863 |         0.023 | 0.04  |
| One-Class SVM (3 cechy)                                      |          0.11  |            0.037 |                         0.031 |      0.89  |         0.031 | 0.052 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii)  |          0.095 |            0.035 |                         0.058 |      0.941 |         0.18  | 0.257 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) |          0.065 |            0.023 |                         0.042 |      0.939 |         0.146 | 0.203 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii)     |          0.135 |            0.052 |                         0.086 |      0.933 |         0.382 | 0.44  |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii)    |          0.095 |            0.035 |                         0.082 |      0.918 |         0.34  | 0.39  |

Poziomy: A impuls = 2 / 3 / 5 / 8 σε; B φ' = 0,5 / 0,3 / 0 / −0,5 (tło φ = 0,7); C Δ = 0,5 / 1 / 2 / 3 σ_x. L = długość anomalii B i C; szok S2-A ma zawsze 10 punktów.

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = 20

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 2% | 1% | 26% | 98% | 8% | 27% | 56% | 85% | 0% | 0% | 12% | 58% |
| PELT (koszt AR) | 3% | 0% | 4% | 85% | 4% | 6% | 32% | 76% | 0% | 0% | 6% | 45% |
| Matrix Profile (reszty AR) | 1% | 4% | 24% | 94% | 7% | 24% | 45% | 74% | 0% | 0% | 7% | 40% |
| cechy okien + kNN (5 cech) | 0% | 10% | 69% | 100% | 3% | 12% | 43% | 78% | 0% | 0% | 7% | 54% |
| cechy okien + kNN (3 cechy) | 0% | 12% | 69% | 100% | 2% | 8% | 30% | 68% | 1% | 0% | 5% | 48% |
| Isolation Forest (5 cech) | 0% | 3% | 12% | 54% | 3% | 4% | 32% | 51% | 0% | 0% | 1% | 4% |
| Isolation Forest (3 cechy) | 1% | 7% | 29% | 66% | 1% | 2% | 15% | 39% | 1% | 0% | 1% | 8% |
| One-Class SVM (5 cech) | 0% | 3% | 19% | 23% | 2% | 3% | 9% | 5% | 0% | 0% | 0% | 0% |
| One-Class SVM (3 cechy) | 0% | 4% | 38% | 50% | 1% | 4% | 11% | 14% | 0% | 0% | 0% | 0% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 1% | 2% | 14% | 39% | 3% | 7% | 37% | 63% | 0% | 0% | 6% | 45% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 1% | 8% | 23% | 50% | 2% | 3% | 24% | 54% | 1% | 0% | 10% | 43% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0% | 10% | 67% | 100% | 3% | 10% | 44% | 76% | 0% | 0% | 11% | 58% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0% | 11% | 70% | 100% | 2% | 5% | 31% | 70% | 0% | 0% | 8% | 53% |

## AUC – rodzaj × poziom intensywności, L = 20

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,58 | 0,55 | 0,63 | 0,75 | 0,53 | 0,70 | 0,80 | 0,84 | 0,54 | 0,62 | 0,91 | 0,98 |
| PELT (koszt AR) | 0,51 | 0,50 | 0,50 | 0,75 | 0,51 | 0,52 | 0,63 | 0,83 | 0,49 | 0,50 | 0,51 | 0,69 |
| Matrix Profile (reszty AR) | 0,60 | 0,72 | 0,91 | 1,00 | 0,72 | 0,86 | 0,93 | 0,96 | 0,53 | 0,57 | 0,77 | 0,95 |
| cechy okien + kNN (5 cech) | 0,60 | 0,70 | 0,98 | 1,00 | 0,65 | 0,83 | 0,92 | 0,98 | 0,52 | 0,58 | 0,87 | 0,99 |
| cechy okien + kNN (3 cechy) | 0,59 | 0,70 | 0,98 | 1,00 | 0,59 | 0,77 | 0,87 | 0,98 | 0,50 | 0,60 | 0,88 | 0,99 |
| Isolation Forest (5 cech) | 0,58 | 0,67 | 0,96 | 1,00 | 0,62 | 0,82 | 0,92 | 0,98 | 0,53 | 0,60 | 0,89 | 1,00 |
| Isolation Forest (3 cechy) | 0,58 | 0,68 | 0,98 | 1,00 | 0,56 | 0,75 | 0,86 | 0,97 | 0,52 | 0,63 | 0,90 | 1,00 |
| One-Class SVM (5 cech) | 0,58 | 0,62 | 0,93 | 0,99 | 0,55 | 0,75 | 0,80 | 0,92 | 0,54 | 0,63 | 0,88 | 0,98 |
| One-Class SVM (3 cechy) | 0,55 | 0,61 | 0,95 | 1,00 | 0,46 | 0,62 | 0,68 | 0,85 | 0,52 | 0,64 | 0,87 | 0,98 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,58 | 0,67 | 0,96 | 1,00 | 0,61 | 0,83 | 0,92 | 0,98 | 0,53 | 0,62 | 0,91 | 1,00 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,58 | 0,69 | 0,97 | 1,00 | 0,56 | 0,76 | 0,87 | 0,97 | 0,52 | 0,63 | 0,91 | 1,00 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,58 | 0,62 | 0,96 | 1,00 | 0,54 | 0,77 | 0,88 | 0,97 | 0,57 | 0,66 | 0,93 | 1,00 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,56 | 0,61 | 0,96 | 1,00 | 0,45 | 0,65 | 0,79 | 0,95 | 0,54 | 0,67 | 0,92 | 0,99 |

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = 50

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 1% | 2% | 26% | 97% | 16% | 67% | 98% | 100% | 1% | 10% | 93% | 100% |
| PELT (koszt AR) | 0% | 1% | 6% | 87% | 1% | 40% | 90% | 100% | 0% | 1% | 66% | 100% |
| Matrix Profile (reszty AR) | 1% | 3% | 27% | 98% | 19% | 77% | 97% | 99% | 0% | 3% | 25% | 86% |
| cechy okien + kNN (5 cech) | 1% | 8% | 61% | 100% | 11% | 66% | 94% | 100% | 0% | 2% | 75% | 100% |
| cechy okien + kNN (3 cechy) | 1% | 8% | 61% | 100% | 5% | 32% | 84% | 100% | 0% | 2% | 84% | 100% |
| Isolation Forest (5 cech) | 1% | 1% | 1% | 18% | 7% | 54% | 90% | 99% | 0% | 0% | 2% | 4% |
| Isolation Forest (3 cechy) | 1% | 3% | 9% | 38% | 2% | 13% | 36% | 57% | 1% | 1% | 11% | 34% |
| One-Class SVM (5 cech) | 1% | 0% | 0% | 0% | 6% | 44% | 64% | 31% | 0% | 0% | 3% | 1% |
| One-Class SVM (3 cechy) | 1% | 1% | 3% | 1% | 3% | 15% | 42% | 43% | 0% | 1% | 29% | 22% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 1% | 3% | 10% | 33% | 6% | 55% | 90% | 89% | 1% | 1% | 15% | 51% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 1% | 5% | 19% | 31% | 2% | 18% | 61% | 81% | 1% | 2% | 22% | 54% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 1% | 6% | 54% | 99% | 9% | 66% | 96% | 100% | 0% | 2% | 75% | 100% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 1% | 7% | 59% | 99% | 3% | 30% | 84% | 100% | 0% | 3% | 86% | 100% |

## AUC – rodzaj × poziom intensywności, L = 50

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,52 | 0,56 | 0,67 | 0,69 | 0,55 | 0,80 | 0,91 | 0,96 | 0,59 | 0,79 | 0,98 | 0,99 |
| PELT (koszt AR) | 0,50 | 0,49 | 0,51 | 0,76 | 0,50 | 0,66 | 0,91 | 0,98 | 0,50 | 0,49 | 0,79 | 0,97 |
| Matrix Profile (reszty AR) | 0,61 | 0,70 | 0,89 | 1,00 | 0,81 | 0,96 | 0,99 | 1,00 | 0,58 | 0,63 | 0,90 | 0,99 |
| cechy okien + kNN (5 cech) | 0,55 | 0,70 | 0,97 | 1,00 | 0,75 | 0,94 | 0,99 | 1,00 | 0,60 | 0,80 | 0,99 | 1,00 |
| cechy okien + kNN (3 cechy) | 0,56 | 0,70 | 0,98 | 1,00 | 0,68 | 0,89 | 0,98 | 1,00 | 0,63 | 0,82 | 0,99 | 1,00 |
| Isolation Forest (5 cech) | 0,55 | 0,67 | 0,96 | 1,00 | 0,75 | 0,94 | 0,99 | 1,00 | 0,60 | 0,80 | 0,98 | 1,00 |
| Isolation Forest (3 cechy) | 0,55 | 0,68 | 0,98 | 1,00 | 0,65 | 0,89 | 0,98 | 1,00 | 0,63 | 0,84 | 0,99 | 1,00 |
| One-Class SVM (5 cech) | 0,51 | 0,60 | 0,87 | 0,99 | 0,67 | 0,89 | 0,92 | 0,94 | 0,62 | 0,80 | 0,95 | 0,96 |
| One-Class SVM (3 cechy) | 0,48 | 0,59 | 0,84 | 0,99 | 0,52 | 0,77 | 0,85 | 0,87 | 0,63 | 0,81 | 0,95 | 0,95 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,55 | 0,67 | 0,96 | 1,00 | 0,75 | 0,95 | 0,99 | 1,00 | 0,61 | 0,83 | 0,96 | 0,99 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,55 | 0,68 | 0,98 | 1,00 | 0,66 | 0,89 | 0,98 | 1,00 | 0,64 | 0,85 | 0,99 | 1,00 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,51 | 0,63 | 0,96 | 1,00 | 0,70 | 0,93 | 0,98 | 1,00 | 0,65 | 0,85 | 0,99 | 1,00 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,48 | 0,61 | 0,96 | 1,00 | 0,55 | 0,82 | 0,96 | 1,00 | 0,66 | 0,86 | 0,99 | 1,00 |

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = 100

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 1% | 2% | 17% | 95% | 43% | 95% | 100% | 100% | 4% | 62% | 100% | 100% |
| PELT (koszt AR) | 3% | 1% | 4% | 80% | 17% | 88% | 100% | 100% | 2% | 18% | 100% | 100% |
| Matrix Profile (reszty AR) | 1% | 5% | 20% | 91% | 58% | 93% | 100% | 100% | 1% | 3% | 35% | 86% |
| cechy okien + kNN (5 cech) | 0% | 5% | 50% | 99% | 34% | 77% | 100% | 100% | 2% | 12% | 90% | 100% |
| cechy okien + kNN (3 cechy) | 0% | 5% | 54% | 99% | 9% | 45% | 96% | 100% | 3% | 19% | 97% | 100% |
| Isolation Forest (5 cech) | 0% | 0% | 0% | 3% | 15% | 62% | 84% | 78% | 0% | 0% | 3% | 2% |
| Isolation Forest (3 cechy) | 0% | 0% | 3% | 17% | 5% | 10% | 27% | 10% | 4% | 2% | 5% | 6% |
| One-Class SVM (5 cech) | 0% | 0% | 0% | 0% | 7% | 27% | 11% | 0% | 1% | 0% | 0% | 0% |
| One-Class SVM (3 cechy) | 1% | 1% | 0% | 0% | 5% | 16% | 19% | 0% | 1% | 0% | 2% | 0% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0% | 1% | 4% | 24% | 23% | 81% | 99% | 97% | 1% | 0% | 9% | 59% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0% | 1% | 11% | 26% | 3% | 35% | 75% | 92% | 3% | 0% | 18% | 62% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0% | 4% | 47% | 99% | 29% | 86% | 100% | 100% | 3% | 10% | 92% | 100% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0% | 4% | 53% | 99% | 9% | 41% | 98% | 100% | 2% | 21% | 100% | 100% |

## AUC – rodzaj × poziom intensywności, L = 100

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,54 | 0,55 | 0,65 | 0,62 | 0,61 | 0,87 | 0,95 | 0,97 | 0,62 | 0,88 | 0,99 | 0,99 |
| PELT (koszt AR) | 0,50 | 0,49 | 0,50 | 0,76 | 0,56 | 0,89 | 0,97 | 0,98 | 0,49 | 0,57 | 0,97 | 0,98 |
| Matrix Profile (reszty AR) | 0,60 | 0,73 | 0,89 | 1,00 | 0,89 | 0,97 | 0,99 | 1,00 | 0,52 | 0,70 | 0,88 | 0,95 |
| cechy okien + kNN (5 cech) | 0,59 | 0,69 | 0,95 | 1,00 | 0,85 | 0,96 | 1,00 | 1,00 | 0,65 | 0,85 | 0,99 | 1,00 |
| cechy okien + kNN (3 cechy) | 0,56 | 0,69 | 0,96 | 1,00 | 0,75 | 0,92 | 0,99 | 1,00 | 0,68 | 0,87 | 0,99 | 1,00 |
| Isolation Forest (5 cech) | 0,59 | 0,70 | 0,94 | 1,00 | 0,84 | 0,96 | 0,99 | 1,00 | 0,64 | 0,82 | 0,98 | 1,00 |
| Isolation Forest (3 cechy) | 0,57 | 0,70 | 0,95 | 1,00 | 0,73 | 0,91 | 0,99 | 1,00 | 0,68 | 0,88 | 0,99 | 1,00 |
| One-Class SVM (5 cech) | 0,56 | 0,57 | 0,81 | 0,99 | 0,73 | 0,84 | 0,87 | 0,90 | 0,65 | 0,78 | 0,89 | 0,92 |
| One-Class SVM (3 cechy) | 0,53 | 0,57 | 0,80 | 0,99 | 0,58 | 0,75 | 0,81 | 0,78 | 0,67 | 0,81 | 0,89 | 0,89 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,59 | 0,69 | 0,93 | 1,00 | 0,85 | 0,97 | 1,00 | 1,00 | 0,66 | 0,86 | 0,96 | 0,98 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,58 | 0,70 | 0,94 | 1,00 | 0,74 | 0,92 | 0,99 | 1,00 | 0,68 | 0,90 | 0,99 | 1,00 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,56 | 0,65 | 0,92 | 1,00 | 0,80 | 0,95 | 0,99 | 1,00 | 0,68 | 0,88 | 1,00 | 1,00 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,53 | 0,63 | 0,92 | 1,00 | 0,61 | 0,86 | 0,98 | 1,00 | 0,71 | 0,90 | 1,00 | 1,00 |

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności (średnio po L)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 1% | 2% | 23% | 97% | 22% | 63% | 85% | 95% | 2% | 24% | 68% | 86% |
| PELT (koszt AR) | 2% | 1% | 5% | 84% | 7% | 45% | 74% | 92% | 1% | 6% | 57% | 82% |
| Matrix Profile (reszty AR) | 1% | 4% | 24% | 94% | 28% | 65% | 81% | 91% | 0% | 2% | 22% | 71% |
| cechy okien + kNN (5 cech) | 0% | 8% | 60% | 100% | 16% | 52% | 79% | 93% | 1% | 5% | 57% | 85% |
| cechy okien + kNN (3 cechy) | 0% | 8% | 61% | 100% | 5% | 28% | 70% | 89% | 1% | 7% | 62% | 83% |
| Isolation Forest (5 cech) | 0% | 1% | 4% | 25% | 8% | 40% | 69% | 76% | 0% | 0% | 2% | 3% |
| Isolation Forest (3 cechy) | 1% | 3% | 14% | 40% | 3% | 8% | 26% | 35% | 2% | 1% | 6% | 16% |
| One-Class SVM (5 cech) | 0% | 1% | 6% | 8% | 5% | 25% | 28% | 12% | 0% | 0% | 1% | 0% |
| One-Class SVM (3 cechy) | 1% | 2% | 14% | 17% | 3% | 12% | 24% | 19% | 0% | 0% | 10% | 7% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 1% | 2% | 9% | 32% | 11% | 48% | 75% | 83% | 1% | 0% | 10% | 52% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 1% | 5% | 18% | 36% | 2% | 19% | 53% | 76% | 2% | 1% | 17% | 53% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0% | 7% | 56% | 99% | 14% | 54% | 80% | 92% | 1% | 4% | 59% | 86% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0% | 7% | 61% | 99% | 5% | 25% | 71% | 90% | 1% | 8% | 65% | 84% |

## AUC – rodzaj × poziom intensywności (średnio po L)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,55 | 0,55 | 0,65 | 0,69 | 0,56 | 0,79 | 0,89 | 0,93 | 0,58 | 0,76 | 0,96 | 0,99 |
| PELT (koszt AR) | 0,50 | 0,49 | 0,50 | 0,75 | 0,52 | 0,69 | 0,84 | 0,93 | 0,50 | 0,52 | 0,76 | 0,88 |
| Matrix Profile (reszty AR) | 0,60 | 0,72 | 0,90 | 1,00 | 0,81 | 0,93 | 0,97 | 0,99 | 0,54 | 0,63 | 0,85 | 0,96 |
| cechy okien + kNN (5 cech) | 0,58 | 0,69 | 0,97 | 1,00 | 0,75 | 0,91 | 0,97 | 0,99 | 0,59 | 0,75 | 0,95 | 1,00 |
| cechy okien + kNN (3 cechy) | 0,57 | 0,70 | 0,97 | 1,00 | 0,67 | 0,86 | 0,94 | 0,99 | 0,60 | 0,76 | 0,95 | 1,00 |
| Isolation Forest (5 cech) | 0,57 | 0,68 | 0,95 | 1,00 | 0,74 | 0,91 | 0,97 | 0,99 | 0,59 | 0,74 | 0,95 | 1,00 |
| Isolation Forest (3 cechy) | 0,57 | 0,69 | 0,97 | 1,00 | 0,65 | 0,85 | 0,94 | 0,99 | 0,61 | 0,78 | 0,96 | 1,00 |
| One-Class SVM (5 cech) | 0,55 | 0,60 | 0,87 | 0,99 | 0,65 | 0,82 | 0,86 | 0,92 | 0,60 | 0,74 | 0,91 | 0,95 |
| One-Class SVM (3 cechy) | 0,52 | 0,59 | 0,86 | 0,99 | 0,52 | 0,71 | 0,78 | 0,83 | 0,60 | 0,75 | 0,90 | 0,94 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,58 | 0,68 | 0,95 | 1,00 | 0,74 | 0,91 | 0,97 | 0,99 | 0,60 | 0,77 | 0,94 | 0,99 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,57 | 0,69 | 0,96 | 1,00 | 0,65 | 0,85 | 0,94 | 0,99 | 0,61 | 0,79 | 0,96 | 1,00 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,55 | 0,63 | 0,94 | 1,00 | 0,68 | 0,88 | 0,95 | 0,99 | 0,63 | 0,80 | 0,97 | 1,00 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,52 | 0,62 | 0,95 | 1,00 | 0,54 | 0,78 | 0,91 | 0,98 | 0,64 | 0,81 | 0,97 | 1,00 |

## Odsetek wykrytych – rodzaj × długość L (średnio po poziomach; dla A długość zawsze 10)

| metoda | A-L20 | A-L50 | A-L100 | B-L20 | B-L50 | B-L100 | C-L20 | C-L50 | C-L100 |
|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 32% | 32% | 29% | 44% | 70% | 84% | 18% | 51% | 66% |
| PELT (koszt AR) | 23% | 24% | 22% | 30% | 58% | 76% | 13% | 42% | 55% |
| Matrix Profile (reszty AR) | 31% | 32% | 29% | 38% | 73% | 88% | 12% | 28% | 31% |
| cechy okien + kNN (5 cech) | 45% | 42% | 38% | 34% | 68% | 78% | 15% | 44% | 51% |
| cechy okien + kNN (3 cechy) | 45% | 42% | 40% | 27% | 55% | 62% | 14% | 46% | 55% |
| Isolation Forest (5 cech) | 17% | 5% | 1% | 22% | 62% | 60% | 1% | 2% | 1% |
| Isolation Forest (3 cechy) | 26% | 13% | 5% | 14% | 27% | 13% | 2% | 12% | 4% |
| One-Class SVM (5 cech) | 11% | 0% | 0% | 5% | 36% | 11% | 0% | 1% | 0% |
| One-Class SVM (3 cechy) | 23% | 2% | 0% | 8% | 26% | 10% | 0% | 13% | 1% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 14% | 12% | 7% | 28% | 60% | 75% | 13% | 17% | 17% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 20% | 14% | 10% | 21% | 40% | 51% | 14% | 20% | 21% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 44% | 40% | 38% | 33% | 68% | 79% | 17% | 44% | 51% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 45% | 42% | 39% | 27% | 54% | 62% | 15% | 47% | 56% |

## Średnio dla rodzaju (wszystkie poziomy i L)

| metoda | A | B | C |
|---|---|---|---|
| reszty AR + CUSUM | 31% | 66% | 45% |
| PELT (koszt AR) | 23% | 55% | 36% |
| Matrix Profile (reszty AR) | 31% | 66% | 24% |
| cechy okien + kNN (5 cech) | 42% | 60% | 37% |
| cechy okien + kNN (3 cechy) | 42% | 48% | 38% |
| Isolation Forest (5 cech) | 8% | 48% | 1% |
| Isolation Forest (3 cechy) | 14% | 18% | 6% |
| One-Class SVM (5 cech) | 4% | 17% | 0% |
| One-Class SVM (3 cechy) | 8% | 14% | 5% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 11% | 54% | 16% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 15% | 38% | 18% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 41% | 60% | 38% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 42% | 48% | 39% |

| metoda | A | B | C |
|---|---|---|---|
| reszty AR + CUSUM | 0,61 | 0,79 | 0,82 |
| PELT (koszt AR) | 0,56 | 0,75 | 0,66 |
| Matrix Profile (reszty AR) | 0,80 | 0,92 | 0,75 |
| cechy okien + kNN (5 cech) | 0,81 | 0,91 | 0,82 |
| cechy okien + kNN (3 cechy) | 0,81 | 0,87 | 0,83 |
| Isolation Forest (5 cech) | 0,80 | 0,90 | 0,82 |
| Isolation Forest (3 cechy) | 0,81 | 0,86 | 0,84 |
| One-Class SVM (5 cech) | 0,75 | 0,81 | 0,80 |
| One-Class SVM (3 cechy) | 0,74 | 0,71 | 0,80 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,80 | 0,90 | 0,82 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,80 | 0,86 | 0,84 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,78 | 0,88 | 0,85 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,77 | 0,80 | 0,85 |

## AP (PR-AUC) średnio dla rodzaju

| metoda | A | B | C |
|---|---|---|---|
| reszty AR + CUSUM | 0,17 | 0,63 | 0,56 |
| PELT (koszt AR) | 0,12 | 0,49 | 0,34 |
| Matrix Profile (reszty AR) | 0,34 | 0,66 | 0,29 |
| cechy okien + kNN (5 cech) | 0,48 | 0,64 | 0,46 |
| cechy okien + kNN (3 cechy) | 0,50 | 0,55 | 0,47 |
| Isolation Forest (5 cech) | 0,37 | 0,62 | 0,38 |
| Isolation Forest (3 cechy) | 0,45 | 0,52 | 0,48 |
| One-Class SVM (5 cech) | 0,15 | 0,40 | 0,23 |
| One-Class SVM (3 cechy) | 0,23 | 0,30 | 0,29 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,32 | 0,63 | 0,33 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,36 | 0,53 | 0,47 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,47 | 0,63 | 0,49 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,48 | 0,52 | 0,51 |

## Długość okna w (poziomy 2 i 3, wszystkie L)

| metoda | A-w20 | A-w50 | A-w100 | B-w20 | B-w50 | B-w100 | C-w20 | C-w50 | C-w100 |
|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile (reszty AR) | 20% | 13% | 6% | 70% | 73% | 65% | 3% | 11% | 15% |
| cechy okien + kNN (5 cech) | 34% | 32% | 26% | 53% | 67% | 60% | 12% | 32% | 33% |
| cechy okien + kNN (3 cechy) | 34% | 33% | 29% | 25% | 46% | 49% | 14% | 36% | 35% |
| Isolation Forest (5 cech) | 14% | 4% | 1% | 49% | 59% | 52% | 1% | 1% | 1% |
| Isolation Forest (3 cechy) | 9% | 6% | 2% | 11% | 16% | 11% | 2% | 3% | 3% |
| One-Class SVM (5 cech) | 22% | 2% | 0% | 42% | 27% | 10% | 3% | 0% | 0% |
| One-Class SVM (3 cechy) | 26% | 6% | 0% | 15% | 16% | 15% | 7% | 6% | 2% |

| metoda | A-w20-L20 | A-w20-L50 | A-w20-L100 | A-w50-L20 | A-w50-L50 | A-w50-L100 | A-w100-L20 | A-w100-L50 | A-w100-L100 | B-w20-L20 | B-w20-L50 | B-w20-L100 | B-w50-L20 | B-w50-L50 | B-w50-L100 | B-w100-L20 | B-w100-L50 | B-w100-L100 | C-w20-L20 | C-w20-L50 | C-w20-L100 | C-w50-L20 | C-w50-L50 | C-w50-L100 | C-w100-L20 | C-w100-L50 | C-w100-L100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile (reszty AR) | 14% | 22% | 23% | 10% | 18% | 11% | 6% | 9% | 4% | 39% | 75% | 95% | 34% | 87% | 98% | 22% | 75% | 99% | 3% | 5% | 2% | 2% | 11% | 20% | 3% | 12% | 31% |
| cechy okien + kNN (5 cech) | 37% | 33% | 31% | 35% | 31% | 29% | 28% | 25% | 26% | 35% | 61% | 64% | 28% | 81% | 93% | 16% | 65% | 100% | 8% | 16% | 11% | 4% | 42% | 51% | 2% | 33% | 63% |
| cechy okien + kNN (3 cechy) | 39% | 33% | 30% | 36% | 31% | 32% | 29% | 28% | 30% | 16% | 30% | 28% | 15% | 54% | 68% | 11% | 51% | 85% | 8% | 20% | 15% | 5% | 47% | 55% | 3% | 34% | 67% |
| Isolation Forest (5 cech) | 19% | 16% | 7% | 7% | 2% | 2% | 2% | 2% | 0% | 26% | 50% | 70% | 21% | 74% | 82% | 9% | 55% | 91% | 2% | 1% | 1% | 3% | 0% | 0% | 0% | 0% | 2% |
| Isolation Forest (3 cechy) | 17% | 7% | 2% | 11% | 4% | 4% | 4% | 1% | 0% | 6% | 10% | 17% | 7% | 21% | 21% | 3% | 14% | 17% | 3% | 2% | 2% | 3% | 5% | 2% | 0% | 1% | 8% |
| One-Class SVM (5 cech) | 35% | 19% | 11% | 7% | 0% | 0% | 0% | 0% | 0% | 26% | 48% | 51% | 10% | 52% | 19% | 1% | 6% | 24% | 6% | 3% | 1% | 0% | 1% | 0% | 0% | 0% | 0% |
| One-Class SVM (3 cechy) | 38% | 24% | 17% | 17% | 1% | 0% | 0% | 0% | 0% | 10% | 13% | 23% | 4% | 22% | 22% | 2% | 13% | 30% | 8% | 8% | 5% | 2% | 15% | 2% | 0% | 0% | 7% |

## Siła zależności tła φ (szok 5σε, φ' = 0, Δ = 2σ_x, L = 50)

| metoda | A-φ0.3 | A-φ0.7 | A-φ0.9 | B-φ0.3 | B-φ0.7 | B-φ0.9 | C-φ0.3 | C-φ0.7 | C-φ0.9 |
|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 33% | 29% | 24% | 12% | 97% | 100% | 100% | 90% | 23% |
| PELT (koszt AR) | 12% | 8% | 7% | 3% | 92% | 100% | 100% | 59% | 3% |
| Matrix Profile (reszty AR) | 29% | 26% | 29% | 5% | 95% | 100% | 99% | 33% | 5% |
| cechy okien + kNN (5 cech) | 68% | 63% | 39% | 5% | 97% | 100% | 100% | 73% | 3% |
| cechy okien + kNN (3 cechy) | 68% | 62% | 38% | 4% | 86% | 100% | 100% | 81% | 4% |
| Isolation Forest (5 cech) | 14% | 4% | 0% | 1% | 88% | 100% | 98% | 1% | 0% |
| Isolation Forest (3 cechy) | 8% | 9% | 1% | 1% | 35% | 100% | 92% | 16% | 1% |
| One-Class SVM (5 cech) | 1% | 1% | 0% | 0% | 62% | 8% | 96% | 2% | 0% |
| One-Class SVM (3 cechy) | 4% | 2% | 0% | 0% | 41% | 26% | 100% | 22% | 0% |

| metoda | A-φ0.3 | A-φ0.7 | A-φ0.9 | B-φ0.3 | B-φ0.7 | B-φ0.9 | C-φ0.3 | C-φ0.7 | C-φ0.9 |
|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,82 | 0,71 | 0,57 | 0,47 | 0,90 | 0,98 | 1,00 | 0,98 | 0,82 |
| PELT (koszt AR) | 0,54 | 0,51 | 0,50 | 0,50 | 0,91 | 0,99 | 0,98 | 0,77 | 0,50 |
| Matrix Profile (reszty AR) | 0,91 | 0,93 | 0,87 | 0,65 | 0,99 | 1,00 | 1,00 | 0,90 | 0,70 |
| cechy okien + kNN (5 cech) | 0,96 | 0,98 | 0,94 | 0,70 | 1,00 | 1,00 | 1,00 | 0,98 | 0,82 |
| cechy okien + kNN (3 cechy) | 0,97 | 0,98 | 0,94 | 0,72 | 0,98 | 1,00 | 1,00 | 0,99 | 0,84 |
| Isolation Forest (5 cech) | 0,96 | 0,97 | 0,91 | 0,72 | 0,99 | 1,00 | 1,00 | 0,98 | 0,85 |
| Isolation Forest (3 cechy) | 0,97 | 0,99 | 0,92 | 0,73 | 0,98 | 1,00 | 1,00 | 0,99 | 0,88 |
| One-Class SVM (5 cech) | 0,90 | 0,89 | 0,82 | 0,67 | 0,93 | 1,00 | 0,99 | 0,94 | 0,81 |
| One-Class SVM (3 cechy) | 0,89 | 0,89 | 0,73 | 0,65 | 0,87 | 0,97 | 0,99 | 0,94 | 0,81 |

Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem) dla każdego φ:

| metoda | 0.3 | 0.7 | 0.9 |
|---|---|---|---|
| reszty AR + CUSUM | 12% | 10% | 12% |
| PELT (koszt AR) | 8% | 7% | 6% |
| Matrix Profile (reszty AR) | 13% | 12% | 12% |
| cechy okien + kNN (5 cech) | 10% | 12% | 10% |
| cechy okien + kNN (3 cechy) | 8% | 10% | 10% |
| Isolation Forest (5 cech) | 12% | 10% | 12% |
| Isolation Forest (3 cechy) | 8% | 9% | 12% |
| One-Class SVM (5 cech) | 8% | 12% | 15% |
| One-Class SVM (3 cechy) | 10% | 11% | 11% |

## Analiza dodatkowa (a) Metody z S1 zastosowane wprost (bez modelu AR), poziomy 2–3, L = 50

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| CUSUM z S1 (bez modelu AR) | 6% | 62% | 38% | 77% | 17% | 96% |
| Matrix Profile | 0% | 4% | 0% | 0% | 7% | 81% |
| kNN z cechami S1 (bez modelu AR) | 1% | 26% | 2% | 29% | 0% | 30% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| CUSUM z S1 (bez modelu AR) | 0,51 | 0,57 | 0,51 | 0,56 | 0,74 | 0,99 |
| Matrix Profile | 0,66 | 0,87 | 0,75 | 0,85 | 0,77 | 0,99 |
| kNN z cechami S1 (bez modelu AR) | 0,62 | 0,86 | 0,88 | 0,98 | 0,81 | 0,99 |

## Analiza dodatkowa (b) Wzrost zależności φ' = 0,95 (B); A i C jak poziomy 2–3, L = 50

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 2% | 30% | 65% | 66% | 10% | 91% |
| PELT (koszt AR) | 2% | 10% | 97% | 98% | 7% | 61% |
| Matrix Profile (reszty AR) | 6% | 30% | 0% | 1% | 3% | 15% |
| cechy okien + kNN (5 cech) | 7% | 59% | 88% | 86% | 2% | 71% |
| cechy okien + kNN (3 cechy) | 7% | 57% | 9% | 7% | 6% | 76% |
| Isolation Forest (5 cech) | 1% | 7% | 70% | 48% | 0% | 2% |
| Isolation Forest (3 cechy) | 1% | 11% | 24% | 8% | 2% | 18% |
| One-Class SVM (5 cech) | 1% | 6% | 95% | 75% | 0% | 9% |
| One-Class SVM (3 cechy) | 4% | 19% | 32% | 17% | 4% | 43% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,58 | 0,64 | 0,99 | 0,99 | 0,81 | 0,98 |
| PELT (koszt AR) | 0,49 | 0,52 | 0,96 | 0,96 | 0,52 | 0,77 |
| Matrix Profile (reszty AR) | 0,73 | 0,92 | 0,03 | 0,03 | 0,63 | 0,85 |
| cechy okien + kNN (5 cech) | 0,72 | 0,97 | 0,97 | 0,96 | 0,81 | 0,98 |
| cechy okien + kNN (3 cechy) | 0,72 | 0,96 | 0,84 | 0,80 | 0,84 | 0,99 |
| Isolation Forest (5 cech) | 0,71 | 0,97 | 0,96 | 0,95 | 0,79 | 0,97 |
| Isolation Forest (3 cechy) | 0,73 | 0,97 | 0,87 | 0,83 | 0,84 | 0,99 |
| One-Class SVM (5 cech) | 0,70 | 0,96 | 0,94 | 0,93 | 0,82 | 0,95 |
| One-Class SVM (3 cechy) | 0,68 | 0,95 | 0,88 | 0,85 | 0,82 | 0,95 |

## Analiza dodatkowa (c) One-Class SVM – szerokość jądra, poziomy 2–3, L = 50

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| One-Class SVM gamma=scale | 2% | 1% | 4% | 3% | 0% | 2% |
| One-Class SVM gamma=0,01 | 1% | 0% | 41% | 61% | 0% | 3% |
| One-Class SVM gamma=0,002 | 1% | 11% | 55% | 87% | 1% | 12% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| One-Class SVM gamma=scale | 0,64 | 0,90 | 0,87 | 0,95 | 0,77 | 0,93 |
| One-Class SVM gamma=0,01 | 0,63 | 0,89 | 0,88 | 0,93 | 0,79 | 0,94 |
| One-Class SVM gamma=0,002 | 0,63 | 0,86 | 0,89 | 0,92 | 0,79 | 0,92 |

Dla porównania – metody S2 z części głównej, poziomy 2–3, L = 50:

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 2% | 26% | 67% | 98% | 10% | 93% |
| PELT (koszt AR) | 1% | 6% | 40% | 90% | 1% | 66% |
| Matrix Profile (reszty AR) | 3% | 27% | 77% | 97% | 3% | 25% |
| cechy okien + kNN (5 cech) | 8% | 61% | 66% | 94% | 2% | 75% |
| cechy okien + kNN (3 cechy) | 8% | 61% | 32% | 84% | 2% | 84% |
| Isolation Forest (5 cech) | 1% | 1% | 54% | 90% | 0% | 2% |
| Isolation Forest (3 cechy) | 3% | 9% | 13% | 36% | 1% | 11% |
| One-Class SVM (5 cech) | 0% | 0% | 44% | 64% | 0% | 3% |
| One-Class SVM (3 cechy) | 1% | 3% | 15% | 42% | 1% | 29% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 3% | 10% | 55% | 90% | 1% | 15% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 5% | 19% | 18% | 61% | 2% | 22% |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 6% | 54% | 66% | 96% | 2% | 75% |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 7% | 59% | 30% | 84% | 3% | 86% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 0,56 | 0,67 | 0,80 | 0,91 | 0,79 | 0,98 |
| PELT (koszt AR) | 0,49 | 0,51 | 0,66 | 0,91 | 0,49 | 0,79 |
| Matrix Profile (reszty AR) | 0,70 | 0,89 | 0,96 | 0,99 | 0,63 | 0,90 |
| cechy okien + kNN (5 cech) | 0,70 | 0,97 | 0,94 | 0,99 | 0,80 | 0,99 |
| cechy okien + kNN (3 cechy) | 0,70 | 0,98 | 0,89 | 0,98 | 0,82 | 0,99 |
| Isolation Forest (5 cech) | 0,67 | 0,96 | 0,94 | 0,99 | 0,80 | 0,98 |
| Isolation Forest (3 cechy) | 0,68 | 0,98 | 0,89 | 0,98 | 0,84 | 0,99 |
| One-Class SVM (5 cech) | 0,60 | 0,87 | 0,89 | 0,92 | 0,80 | 0,95 |
| One-Class SVM (3 cechy) | 0,59 | 0,84 | 0,77 | 0,85 | 0,81 | 0,95 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,67 | 0,96 | 0,95 | 0,99 | 0,83 | 0,96 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,68 | 0,98 | 0,89 | 0,98 | 0,85 | 0,99 |
| One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 0,63 | 0,96 | 0,93 | 0,98 | 0,85 | 0,99 |
| One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 0,61 | 0,96 | 0,82 | 0,96 | 0,86 | 0,99 |

Fałszywe alarmy na walidacji w analizach dodatkowych (odsetek szeregów z ≥1 alarmem):

|                                                            |   odsetek |
|:-----------------------------------------------------------|----------:|
| ('a_metody_S1_wprost', 'CUSUM z S1 (bez modelu AR)')       |     0.07  |
| ('a_metody_S1_wprost', 'Matrix Profile')                   |     0.095 |
| ('a_metody_S1_wprost', 'kNN z cechami S1 (bez modelu AR)') |     0.07  |
| ('b_wzrost_phi_0.95', 'Isolation Forest (3 cechy)')        |     0.09  |
| ('b_wzrost_phi_0.95', 'Isolation Forest (5 cech)')         |     0.105 |
| ('b_wzrost_phi_0.95', 'Matrix Profile (reszty AR)')        |     0.12  |
| ('b_wzrost_phi_0.95', 'One-Class SVM (3 cechy)')           |     0.11  |
| ('b_wzrost_phi_0.95', 'One-Class SVM (5 cech)')            |     0.125 |
| ('b_wzrost_phi_0.95', 'PELT (koszt AR)')                   |     0.07  |
| ('b_wzrost_phi_0.95', 'cechy okien + kNN (3 cechy)')       |     0.105 |
| ('b_wzrost_phi_0.95', 'cechy okien + kNN (5 cech)')        |     0.115 |
| ('b_wzrost_phi_0.95', 'reszty AR + CUSUM')                 |     0.105 |
| ('c_gamma_OCSVM', 'One-Class SVM gamma=0,002')             |     0.145 |
| ('c_gamma_OCSVM', 'One-Class SVM gamma=0,01')              |     0.125 |
| ('c_gamma_OCSVM', 'One-Class SVM gamma=scale')             |     0.11  |

## Analiza dodatkowa (d) CUSUM mieszany – wariant zaproponowany po wynikach (a); poziomy 1–4, L = 50

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 1% | 0% | 24% | 100% | 26% | 77% | 97% | 100% | 2% | 14% | 94% | 100% |
| CUSUM z S1 (bez modelu AR) | 1% | 7% | 58% | 100% | 16% | 40% | 80% | 95% | 2% | 16% | 95% | 100% |
| CUSUM mieszany (poziom na x, reszta na resztach AR) | 1% | 0% | 20% | 100% | 25% | 76% | 96% | 100% | 3% | 14% | 91% | 100% |

| metoda | A | B | C |
|---|---|---|---|
| reszty AR + CUSUM | 31% | 75% | 52% |
| CUSUM z S1 (bez modelu AR) | 42% | 58% | 53% |
| CUSUM mieszany (poziom na x, reszta na resztach AR) | 30% | 74% | 52% |

| metoda | A | B | C |
|---|---|---|---|
| reszty AR + CUSUM | 0,61 | 0,81 | 0,84 |
| CUSUM z S1 (bez modelu AR) | 0,55 | 0,56 | 0,81 |
| CUSUM mieszany (poziom na x, reszta na resztach AR) | 0,63 | 0,81 | 0,85 |

Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem):

| metoda                                              |   odsetek |
|:----------------------------------------------------|----------:|
| CUSUM mieszany (poziom na x, reszta na resztach AR) |     0.1   |
| CUSUM z S1 (bez modelu AR)                          |     0.07  |
| reszty AR + CUSUM                                   |     0.105 |
