# S3 – tabele wyników (generowane automatycznie przez src/ewaluacja/wykresy_s3.py)

Część główna: 1200 szeregów z anomaliami (3600 anomalii na metodę), 200 szeregów walidacyjnych bez anomalii. Próg: fałszywy alarm w 10% szeregów kalibracyjnych bez anomalii.

## Kalibracja

| metoda                                                       |     prog |   kara_pelt |
|:-------------------------------------------------------------|---------:|------------:|
| STL + EWMA                                                   | 0.718593 |         nan |
| PELT (reszty STL)                                            | 0        |          13 |
| Matrix Profile z normalizacją                                | 4.11751  |         nan |
| cechy cykli + kNN (5 cech)                                   | 3.1501   |         nan |
| cechy cykli + kNN (3 cechy)                                  | 2.15551  |         nan |
| Isolation Forest (5 cech)                                    | 0.68457  |         nan |
| Isolation Forest (3 cechy)                                   | 0.690829 |         nan |
| autoenkoder                                                  | 0.316618 |         nan |
| Matrix Profile bez normalizacji                              | 9.75571  |         nan |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii)  | 0.679145 |         nan |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0.686936 |         nan |
| autoenkoder (uczony na szeregach bez anomalii)               | 0.311671 |         nan |

## Fałszywe alarmy (walidacja na nowych szeregach bez anomalii) i miary punktowe

| metoda                                                       |   odsetek_z_fa |   fa_na_1000_pkt |   fa_w_szeregach_z_anomaliami |   precyzja |   czulosc_pkt |    f1 |
|:-------------------------------------------------------------|---------------:|-----------------:|------------------------------:|-----------:|--------------:|------:|
| STL + EWMA                                                   |          0.135 |            0.052 |                         0.138 |      0.987 |         0.549 | 0.653 |
| PELT (reszty STL)                                            |          0.065 |            0.023 |                         0.05  |      0.934 |         0.675 | 0.73  |
| Matrix Profile z normalizacją                                |          0.135 |            0.078 |                         0.178 |      0.988 |         0.442 | 0.553 |
| cechy cykli + kNN (5 cech)                                   |          0.155 |            0.062 |                         0.032 |      0.996 |         0.731 | 0.8   |
| cechy cykli + kNN (3 cechy)                                  |          0.13  |            0.048 |                         0.022 |      0.997 |         0.712 | 0.786 |
| Isolation Forest (5 cech)                                    |          0.13  |            0.045 |                         0     |      1     |         0.18  | 0.274 |
| Isolation Forest (3 cechy)                                   |          0.12  |            0.043 |                         0.001 |      0.999 |         0.136 | 0.215 |
| autoenkoder                                                  |          0.165 |            0.072 |                         0.144 |      0.973 |         0.386 | 0.502 |
| Matrix Profile bez normalizacji                              |          0.135 |            0.077 |                         0.155 |      0.982 |         0.311 | 0.402 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii)  |          0.11  |            0.047 |                         0.012 |      0.997 |         0.506 | 0.608 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) |          0.13  |            0.057 |                         0.022 |      0.995 |         0.445 | 0.552 |
| autoenkoder (uczony na szeregach bez anomalii)               |          0.13  |            0.057 |                         0.123 |      0.975 |         0.36  | 0.481 |

Poziomy: A Δθ = π/8 / π/4 / π/2 / π; B k = 0,75 / 0,5 / 0,25 / 0 (amplituda k·A); C r = 0,9 / 0,75 / 0,6 / 0,5 (okres r·P).

## Odsetek wykrytych i AUC – poziom intensywności przy L = 50 (1 cykl/e)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 12% | 97% | 100% | 100% | 2% | 42% | 94% | 100% | 13% | 100% | 100% | 100% |
| PELT (reszty STL) | 19% | 100% | 100% | 100% | 3% | 57% | 100% | 100% | 18% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 46% | 100% | 100% | 24% | 96% | 100% | 100% | 3% | 100% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 64% | 100% | 100% | 100% | 33% | 100% | 100% | 100% | 50% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 69% | 100% | 100% | 100% | 42% | 100% | 100% | 100% | 47% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 25% | 19% | 97% | 100% | 17% | 17% | 24% | 92% | 27% | 100% | 100% | 98% |
| Isolation Forest (3 cechy) | 23% | 15% | 78% | 100% | 19% | 48% | 51% | 96% | 23% | 100% | 100% | 95% |
| autoenkoder | 7% | 65% | 100% | 100% | 4% | 58% | 97% | 100% | 12% | 100% | 100% | 100% |
| Matrix Profile bez normalizacji | 3% | 64% | 100% | 100% | 2% | 73% | 100% | 100% | 9% | 100% | 100% | 100% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 15% | 94% | 100% | 100% | 22% | 90% | 100% | 100% | 30% | 100% | 100% | 100% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 23% | 75% | 100% | 100% | 25% | 84% | 100% | 100% | 24% | 100% | 100% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 4% | 57% | 100% | 100% | 2% | 9% | 63% | 99% | 11% | 100% | 100% | 100% |

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 0,80 | 0,96 | 0,99 | 0,99 | 0,64 | 0,88 | 0,96 | 0,97 | 0,75 | 0,91 | 0,94 | 0,91 |
| PELT (reszty STL) | 0,54 | 0,91 | 0,95 | 0,97 | 0,50 | 0,68 | 0,92 | 0,93 | 0,54 | 0,84 | 0,89 | 0,85 |
| Matrix Profile z normalizacją | 0,61 | 0,77 | 0,94 | 0,96 | 0,83 | 0,99 | 1,00 | 1,00 | 0,64 | 0,96 | 1,00 | 1,00 |
| cechy cykli + kNN (5 cech) | 0,97 | 1,00 | 1,00 | 1,00 | 0,90 | 1,00 | 1,00 | 1,00 | 0,93 | 1,00 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 0,98 | 1,00 | 1,00 | 1,00 | 0,92 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 0,96 | 1,00 | 1,00 | 1,00 | 0,90 | 1,00 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 0,97 | 1,00 | 1,00 | 1,00 | 0,92 | 1,00 | 1,00 | 1,00 | 0,93 | 1,00 | 1,00 | 1,00 |
| autoenkoder | 0,65 | 0,85 | 0,97 | 0,98 | 0,60 | 0,83 | 0,95 | 0,98 | 0,69 | 0,99 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,63 | 0,82 | 0,94 | 0,96 | 0,67 | 0,96 | 1,00 | 1,00 | 0,66 | 0,98 | 1,00 | 1,00 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,96 | 1,00 | 1,00 | 1,00 | 0,89 | 1,00 | 1,00 | 1,00 | 0,90 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,96 | 0,99 | 1,00 | 1,00 | 0,90 | 0,99 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,65 | 0,85 | 0,96 | 0,98 | 0,55 | 0,72 | 0,87 | 0,92 | 0,69 | 0,99 | 1,00 | 1,00 |

## Odsetek wykrytych i AUC – poziom intensywności przy L = 100 (2 cykl/e)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 31% | 100% | 100% | 100% | 6% | 75% | 100% | 100% | 100% | 100% | 100% | 100% |
| PELT (reszty STL) | 35% | 100% | 100% | 100% | 4% | 92% | 100% | 100% | 100% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 44% | 100% | 100% | 46% | 100% | 100% | 100% | 82% | 100% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 85% | 100% | 100% | 100% | 56% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 91% | 100% | 100% | 100% | 64% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 0% | 4% | 99% | 0% | 0% | 23% | 85% | 100% | 100% | 100% | 16% |
| Isolation Forest (3 cechy) | 0% | 0% | 12% | 71% | 6% | 0% | 33% | 95% | 99% | 100% | 100% | 22% |
| autoenkoder | 5% | 66% | 100% | 100% | 7% | 65% | 95% | 99% | 99% | 100% | 100% | 100% |
| Matrix Profile bez normalizacji | 2% | 52% | 100% | 100% | 10% | 78% | 100% | 100% | 96% | 100% | 100% | 100% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 12% | 92% | 100% | 100% | 27% | 96% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 22% | 74% | 100% | 100% | 34% | 94% | 100% | 100% | 92% | 100% | 100% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 4% | 63% | 100% | 100% | 1% | 20% | 74% | 99% | 96% | 100% | 100% | 100% |

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 0,86 | 0,98 | 0,99 | 1,00 | 0,68 | 0,93 | 0,98 | 0,99 | 0,88 | 0,96 | 0,98 | 0,96 |
| PELT (reszty STL) | 0,58 | 0,94 | 0,96 | 0,97 | 0,49 | 0,82 | 0,95 | 0,95 | 0,83 | 0,92 | 0,88 | 0,87 |
| Matrix Profile z normalizacją | 0,53 | 0,63 | 0,71 | 0,73 | 0,87 | 0,99 | 1,00 | 1,00 | 0,71 | 0,96 | 0,97 | 0,97 |
| cechy cykli + kNN (5 cech) | 0,99 | 1,00 | 1,00 | 1,00 | 0,92 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 0,99 | 1,00 | 1,00 | 1,00 | 0,93 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 0,97 | 1,00 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 0,97 | 1,00 | 1,00 | 1,00 | 0,93 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 |
| autoenkoder | 0,55 | 0,67 | 0,73 | 0,75 | 0,58 | 0,73 | 0,77 | 0,72 | 0,74 | 1,00 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,54 | 0,66 | 0,71 | 0,74 | 0,69 | 0,95 | 0,96 | 0,81 | 0,72 | 0,96 | 0,97 | 0,97 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,98 | 1,00 | 1,00 | 1,00 | 0,90 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,98 | 0,99 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,55 | 0,67 | 0,72 | 0,75 | 0,51 | 0,62 | 0,72 | 0,69 | 0,74 | 1,00 | 1,00 | 1,00 |

## Odsetek wykrytych i AUC – poziom intensywności przy L = 200 (4 cykl/e)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 40% | 100% | 100% | 100% | 6% | 77% | 100% | 100% | 100% | 100% | 100% | 100% |
| PELT (reszty STL) | 51% | 100% | 100% | 100% | 9% | 98% | 100% | 100% | 100% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 39% | 100% | 100% | 81% | 100% | 100% | 100% | 100% | 99% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 42% | 98% | 100% | 100% | 49% | 94% | 100% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 41% | 99% | 100% | 100% | 46% | 94% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 0% | 0% | 33% | 0% | 0% | 17% | 23% | 100% | 100% | 100% | 0% |
| Isolation Forest (3 cechy) | 0% | 0% | 2% | 6% | 0% | 2% | 36% | 83% | 100% | 99% | 82% | 0% |
| autoenkoder | 4% | 73% | 100% | 100% | 12% | 45% | 70% | 98% | 100% | 100% | 100% | 100% |
| Matrix Profile bez normalizacji | 3% | 54% | 100% | 100% | 16% | 57% | 99% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 12% | 85% | 100% | 100% | 45% | 98% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 35% | 62% | 100% | 100% | 58% | 96% | 100% | 100% | 100% | 100% | 100% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 4% | 72% | 100% | 100% | 2% | 18% | 71% | 99% | 100% | 100% | 100% | 100% |

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 0,89 | 0,98 | 1,00 | 1,00 | 0,69 | 0,94 | 0,98 | 0,99 | 0,94 | 0,98 | 0,99 | 0,98 |
| PELT (reszty STL) | 0,60 | 0,96 | 0,95 | 0,96 | 0,50 | 0,84 | 0,96 | 0,96 | 0,88 | 0,91 | 0,93 | 0,86 |
| Matrix Profile z normalizacją | 0,54 | 0,56 | 0,61 | 0,61 | 0,90 | 1,00 | 1,00 | 1,00 | 0,66 | 0,91 | 0,90 | 0,88 |
| cechy cykli + kNN (5 cech) | 0,98 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 | 0,98 | 1,00 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 0,98 | 0,99 | 0,99 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 | 0,98 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 0,93 | 1,00 | 1,00 | 1,00 | 0,93 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 0,93 | 0,99 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |
| autoenkoder | 0,56 | 0,59 | 0,63 | 0,61 | 0,58 | 0,62 | 0,64 | 0,61 | 0,71 | 1,00 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,54 | 0,57 | 0,60 | 0,61 | 0,69 | 0,84 | 0,82 | 0,61 | 0,68 | 0,89 | 0,89 | 0,88 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,98 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,98 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,56 | 0,58 | 0,62 | 0,61 | 0,52 | 0,58 | 0,66 | 0,62 | 0,73 | 1,00 | 1,00 | 1,00 |

## Odsetek wykrytych anomalii i AUC – rodzaj × poziom intensywności (średnio po L)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 28% | 99% | 100% | 100% | 5% | 65% | 98% | 100% | 71% | 100% | 100% | 100% |
| PELT (reszty STL) | 35% | 100% | 100% | 100% | 5% | 82% | 100% | 100% | 73% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 43% | 100% | 100% | 50% | 99% | 100% | 100% | 62% | 100% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 64% | 99% | 100% | 100% | 46% | 98% | 100% | 100% | 83% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 67% | 100% | 100% | 100% | 51% | 98% | 100% | 100% | 82% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 8% | 6% | 34% | 77% | 6% | 6% | 21% | 67% | 76% | 100% | 100% | 38% |
| Isolation Forest (3 cechy) | 8% | 5% | 31% | 59% | 8% | 17% | 40% | 91% | 74% | 100% | 94% | 39% |
| autoenkoder | 5% | 68% | 100% | 100% | 8% | 56% | 87% | 99% | 70% | 100% | 100% | 100% |
| Matrix Profile bez normalizacji | 3% | 57% | 100% | 100% | 9% | 69% | 100% | 100% | 68% | 100% | 100% | 100% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 13% | 90% | 100% | 100% | 31% | 95% | 100% | 100% | 77% | 100% | 100% | 100% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 27% | 70% | 100% | 100% | 39% | 91% | 100% | 100% | 72% | 100% | 100% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 4% | 64% | 100% | 100% | 2% | 16% | 69% | 99% | 69% | 100% | 100% | 100% |

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 0,85 | 0,97 | 0,99 | 0,99 | 0,67 | 0,92 | 0,97 | 0,98 | 0,86 | 0,95 | 0,97 | 0,95 |
| PELT (reszty STL) | 0,58 | 0,94 | 0,95 | 0,97 | 0,50 | 0,78 | 0,94 | 0,95 | 0,75 | 0,89 | 0,90 | 0,86 |
| Matrix Profile z normalizacją | 0,56 | 0,65 | 0,75 | 0,77 | 0,87 | 0,99 | 1,00 | 1,00 | 0,67 | 0,94 | 0,96 | 0,95 |
| cechy cykli + kNN (5 cech) | 0,98 | 1,00 | 1,00 | 1,00 | 0,93 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 0,99 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 0,95 | 1,00 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 0,96 | 0,99 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 |
| autoenkoder | 0,59 | 0,70 | 0,77 | 0,78 | 0,59 | 0,73 | 0,79 | 0,77 | 0,71 | 0,99 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,57 | 0,68 | 0,75 | 0,77 | 0,68 | 0,92 | 0,93 | 0,81 | 0,68 | 0,94 | 0,95 | 0,95 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,97 | 1,00 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,98 | 1,00 | 1,00 | 1,00 | 0,92 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,59 | 0,70 | 0,77 | 0,78 | 0,53 | 0,64 | 0,75 | 0,75 | 0,72 | 0,99 | 1,00 | 1,00 |

## Odsetek wykrytych i AUC – rodzaj × długość L (średnio po poziomach)

| metoda | A-L50 | A-L100 | A-L200 | B-L50 | B-L100 | B-L200 | C-L50 | C-L100 | C-L200 |
|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 77% | 83% | 85% | 60% | 70% | 71% | 78% | 100% | 100% |
| PELT (reszty STL) | 80% | 84% | 88% | 65% | 74% | 77% | 80% | 100% | 100% |
| Matrix Profile z normalizacją | 62% | 62% | 60% | 80% | 86% | 95% | 76% | 96% | 100% |
| cechy cykli + kNN (5 cech) | 91% | 96% | 85% | 83% | 89% | 86% | 88% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 92% | 98% | 85% | 86% | 91% | 85% | 87% | 100% | 100% |
| Isolation Forest (5 cech) | 60% | 26% | 8% | 38% | 27% | 10% | 81% | 79% | 75% |
| Isolation Forest (3 cechy) | 54% | 21% | 2% | 54% | 34% | 30% | 80% | 80% | 70% |
| autoenkoder | 68% | 68% | 69% | 65% | 66% | 56% | 78% | 100% | 100% |
| Matrix Profile bez normalizacji | 67% | 64% | 64% | 69% | 72% | 68% | 77% | 99% | 100% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 77% | 76% | 74% | 78% | 81% | 86% | 82% | 100% | 100% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 74% | 74% | 74% | 77% | 82% | 88% | 81% | 98% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 65% | 67% | 69% | 43% | 48% | 48% | 78% | 99% | 100% |

| metoda | A-L50 | A-L100 | A-L200 | B-L50 | B-L100 | B-L200 | C-L50 | C-L100 | C-L200 |
|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 0,93 | 0,96 | 0,97 | 0,86 | 0,89 | 0,90 | 0,88 | 0,94 | 0,97 |
| PELT (reszty STL) | 0,85 | 0,86 | 0,87 | 0,76 | 0,80 | 0,81 | 0,78 | 0,87 | 0,90 |
| Matrix Profile z normalizacją | 0,82 | 0,65 | 0,58 | 0,95 | 0,96 | 0,97 | 0,90 | 0,90 | 0,84 |
| cechy cykli + kNN (5 cech) | 0,99 | 1,00 | 0,99 | 0,98 | 0,98 | 0,99 | 0,98 | 0,99 | 1,00 |
| cechy cykli + kNN (3 cechy) | 0,99 | 1,00 | 0,99 | 0,98 | 0,98 | 0,99 | 0,98 | 0,99 | 1,00 |
| Isolation Forest (5 cech) | 0,99 | 0,99 | 0,98 | 0,97 | 0,98 | 0,98 | 0,98 | 0,99 | 0,99 |
| Isolation Forest (3 cechy) | 0,99 | 0,99 | 0,98 | 0,98 | 0,98 | 0,99 | 0,98 | 0,99 | 0,99 |
| autoenkoder | 0,86 | 0,68 | 0,60 | 0,84 | 0,70 | 0,61 | 0,92 | 0,93 | 0,93 |
| Matrix Profile bez normalizacji | 0,84 | 0,66 | 0,58 | 0,91 | 0,85 | 0,74 | 0,91 | 0,90 | 0,84 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,99 | 0,99 | 1,00 | 0,97 | 0,98 | 0,98 | 0,97 | 0,99 | 0,99 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,99 | 0,99 | 0,99 | 0,97 | 0,98 | 0,99 | 0,98 | 0,99 | 0,99 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,86 | 0,67 | 0,59 | 0,76 | 0,64 | 0,59 | 0,92 | 0,93 | 0,93 |

## Średnio dla rodzaju (wszystkie poziomy i L)

| metoda | A | B | C |
|---|---|---|---|
| STL + EWMA | 82% | 67% | 93% |
| PELT (reszty STL) | 84% | 72% | 93% |
| Matrix Profile z normalizacją | 61% | 87% | 90% |
| cechy cykli + kNN (5 cech) | 91% | 86% | 96% |
| cechy cykli + kNN (3 cechy) | 92% | 87% | 96% |
| Isolation Forest (5 cech) | 31% | 25% | 78% |
| Isolation Forest (3 cechy) | 26% | 39% | 77% |
| autoenkoder | 68% | 62% | 93% |
| Matrix Profile bez normalizacji | 65% | 70% | 92% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 76% | 82% | 94% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 74% | 83% | 93% |
| autoenkoder (uczony na szeregach bez anomalii) | 67% | 46% | 92% |

| metoda | A | B | C |
|---|---|---|---|
| STL + EWMA | 0,95 | 0,89 | 0,93 |
| PELT (reszty STL) | 0,86 | 0,79 | 0,85 |
| Matrix Profile z normalizacją | 0,68 | 0,96 | 0,88 |
| cechy cykli + kNN (5 cech) | 0,99 | 0,98 | 0,99 |
| cechy cykli + kNN (3 cechy) | 0,99 | 0,98 | 0,99 |
| Isolation Forest (5 cech) | 0,99 | 0,98 | 0,99 |
| Isolation Forest (3 cechy) | 0,99 | 0,98 | 0,99 |
| autoenkoder | 0,71 | 0,72 | 0,93 |
| Matrix Profile bez normalizacji | 0,69 | 0,83 | 0,88 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,99 | 0,98 | 0,99 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,99 | 0,98 | 0,99 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,71 | 0,67 | 0,93 |

## AP (PR-AUC) średnio dla rodzaju

| metoda | A | B | C |
|---|---|---|---|
| STL + EWMA | 0,80 | 0,64 | 0,79 |
| PELT (reszty STL) | 0,73 | 0,60 | 0,71 |
| Matrix Profile z normalizacją | 0,34 | 0,85 | 0,66 |
| cechy cykli + kNN (5 cech) | 0,95 | 0,90 | 0,95 |
| cechy cykli + kNN (3 cechy) | 0,95 | 0,91 | 0,95 |
| Isolation Forest (5 cech) | 0,90 | 0,87 | 0,94 |
| Isolation Forest (3 cechy) | 0,88 | 0,90 | 0,93 |
| autoenkoder | 0,39 | 0,32 | 0,81 |
| Matrix Profile bez normalizacji | 0,36 | 0,54 | 0,67 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,94 | 0,89 | 0,94 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,87 | 0,88 | 0,92 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,38 | 0,25 | 0,81 |

## Długość okna w (poziomy 2 i 3, wszystkie L)

| metoda | A-w25 | A-w50 | A-w100 | B-w25 | B-w50 | B-w100 | C-w25 | C-w50 | C-w100 |
|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile z normalizacją | 41% | 74% | 98% | 78% | 100% | 100% | 39% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 2% | 100% | 100% | 1% | 99% | 99% | 47% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 1% | 99% | 100% | 0% | 99% | 99% | 2% | 100% | 100% |
| Isolation Forest (5 cech) | 11% | 19% | 31% | 6% | 12% | 1% | 37% | 100% | 100% |
| Isolation Forest (3 cechy) | 18% | 14% | 11% | 22% | 28% | 1% | 28% | 97% | 99% |
| autoenkoder | 53% | 87% | 100% | 10% | 69% | 81% | 97% | 100% | 100% |
| Matrix Profile bez normalizacji | 56% | 80% | 98% | 60% | 86% | 94% | 97% | 100% | 100% |

| metoda | A-w25 | A-w50 | A-w100 | B-w25 | B-w50 | B-w100 | C-w25 | C-w50 | C-w100 |
|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile z normalizacją | 0,54 | 0,70 | 0,86 | 0,88 | 0,99 | 1,00 | 0,51 | 0,94 | 0,97 |
| cechy cykli + kNN (5 cech) | 0,99 | 1,00 | 1,00 | 0,97 | 1,00 | 0,99 | 0,98 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 0,98 | 1,00 | 1,00 | 0,98 | 1,00 | 1,00 | 0,98 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 0,97 | 1,00 | 1,00 | 0,96 | 1,00 | 0,99 | 0,97 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 0,97 | 1,00 | 1,00 | 0,97 | 1,00 | 0,99 | 0,97 | 1,00 | 1,00 |
| autoenkoder | 0,60 | 0,74 | 0,88 | 0,62 | 0,75 | 0,84 | 0,87 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,58 | 0,72 | 0,87 | 0,86 | 0,92 | 0,95 | 0,80 | 0,94 | 0,97 |

| metoda | A-w25-L50 | A-w25-L100 | A-w25-L200 | A-w50-L50 | A-w50-L100 | A-w50-L200 | A-w100-L50 | A-w100-L100 | A-w100-L200 | B-w25-L50 | B-w25-L100 | B-w25-L200 | B-w50-L50 | B-w50-L100 | B-w50-L200 | B-w100-L50 | B-w100-L100 | B-w100-L200 | C-w25-L50 | C-w25-L100 | C-w25-L200 | C-w50-L50 | C-w50-L100 | C-w50-L200 | C-w100-L50 | C-w100-L100 | C-w100-L200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile z normalizacją | 44% | 41% | 37% | 74% | 74% | 74% | 97% | 98% | 99% | 66% | 76% | 92% | 99% | 100% | 100% | 99% | 100% | 100% | 44% | 32% | 42% | 100% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 5% | 0% | 0% | 100% | 100% | 99% | 100% | 100% | 100% | 1% | 1% | 0% | 100% | 100% | 97% | 97% | 100% | 100% | 55% | 86% | 1% | 100% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 2% | 0% | 0% | 100% | 100% | 98% | 100% | 100% | 100% | 1% | 0% | 0% | 100% | 100% | 97% | 96% | 100% | 100% | 7% | 0% | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 28% | 5% | 0% | 57% | 1% | 0% | 42% | 50% | 0% | 12% | 5% | 2% | 24% | 8% | 4% | 0% | 0% | 3% | 74% | 11% | 26% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (3 cechy) | 33% | 15% | 5% | 41% | 2% | 0% | 30% | 2% | 0% | 28% | 23% | 14% | 48% | 17% | 18% | 0% | 1% | 3% | 51% | 10% | 24% | 100% | 100% | 91% | 96% | 100% | 100% |
| autoenkoder | 52% | 56% | 52% | 85% | 88% | 88% | 99% | 100% | 100% | 13% | 8% | 10% | 78% | 77% | 52% | 81% | 85% | 77% | 94% | 100% | 97% | 100% | 100% | 100% | 100% | 100% | 100% |
| Matrix Profile bez normalizacji | 57% | 56% | 55% | 77% | 80% | 83% | 97% | 99% | 99% | 75% | 57% | 47% | 84% | 89% | 86% | 86% | 97% | 98% | 95% | 98% | 97% | 100% | 100% | 100% | 100% | 100% | 100% |

## Amplituda cyklu tła A przy σ = 1 (poziom 3: faza +π/2, amplituda ×0,25, okres ×0,6; L = 100)

| metoda | A-A1.0 | A-A3.0 | A-A5.0 | B-A1.0 | B-A3.0 | B-A5.0 | C-A1.0 | C-A3.0 | C-A5.0 |
|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 63% | 100% | 100% | 3% | 100% | 100% | 92% | 100% | 100% |
| PELT (reszty STL) | 84% | 100% | 100% | 1% | 100% | 100% | 96% | 100% | 100% |
| Matrix Profile z normalizacją | 25% | 100% | 100% | 91% | 100% | 100% | 53% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 0% | 100% | 100% | 11% | 100% | 100% | 3% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 0% | 100% | 100% | 7% | 100% | 100% | 0% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 8% | 34% | 2% | 19% | 2% | 1% | 100% | 100% |
| Isolation Forest (3 cechy) | 0% | 10% | 20% | 6% | 35% | 16% | 0% | 100% | 100% |
| autoenkoder | 13% | 100% | 100% | 2% | 92% | 100% | 45% | 100% | 100% |
| Matrix Profile bez normalizacji | 9% | 100% | 100% | 2% | 100% | 100% | 27% | 100% | 100% |

| metoda | A-A1.0 | A-A3.0 | A-A5.0 | B-A1.0 | B-A3.0 | B-A5.0 | C-A1.0 | C-A3.0 | C-A5.0 |
|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 0,92 | 0,99 | 1,00 | 0,73 | 0,97 | 0,99 | 0,87 | 0,97 | 0,98 |
| PELT (reszty STL) | 0,75 | 0,95 | 0,99 | 0,49 | 0,95 | 0,95 | 0,65 | 0,88 | 0,92 |
| Matrix Profile z normalizacją | 0,63 | 0,73 | 0,75 | 0,95 | 1,00 | 1,00 | 0,88 | 0,97 | 0,98 |
| cechy cykli + kNN (5 cech) | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 0,99 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 0,99 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 |
| autoenkoder | 0,59 | 0,73 | 0,78 | 0,55 | 0,76 | 0,95 | 0,87 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,59 | 0,73 | 0,74 | 0,63 | 0,96 | 0,98 | 0,81 | 0,96 | 0,97 |

Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem) dla każdego A:

| metoda | 1.0 | 3.0 | 5.0 |
|---|---|---|---|
| STL + EWMA | 14% | 14% | 14% |
| PELT (reszty STL) | 6% | 6% | 6% |
| Matrix Profile z normalizacją | 14% | 14% | 18% |
| cechy cykli + kNN (5 cech) | 12% | 16% | 15% |
| cechy cykli + kNN (3 cechy) | 12% | 13% | 14% |
| Isolation Forest (5 cech) | 10% | 13% | 15% |
| Isolation Forest (3 cechy) | 10% | 12% | 12% |
| autoenkoder | 14% | 16% | 16% |
| Matrix Profile bez normalizacji | 12% | 14% | 14% |

Kara PELT dla każdego A:

|   A |   kara_pelt |
|----:|------------:|
|   1 |          13 |
|   3 |          13 |
|   5 |          13 |

## Analiza dodatkowa (a) Model cyklu: STL z oknem 7 cykli / STL z oknem dłuższym niż szereg / sinus dopasowany globalnie (poziomy 2–3)

| metoda | A-L50 | A-L100 | A-L200 | B-L50 | B-L100 | B-L200 | C-L50 | C-L100 | C-L200 |
|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA [STL, okno sezonowości 7 cykli] | 99% | 90% | 59% | 54% | 41% | 0% | 100% | 100% | 100% |
| STL + EWMA | 99% | 100% | 100% | 67% | 87% | 87% | 100% | 100% | 100% |
| STL + EWMA [model: sinus dopasowany globalnie] | 100% | 100% | 100% | 81% | 86% | 78% | 100% | 100% | 100% |
| PELT (reszty STL) [STL, okno sezonowości 7 cykli] | 66% | 56% | 49% | 10% | 9% | 0% | 100% | 100% | 100% |
| PELT (reszty STL) | 100% | 100% | 100% | 76% | 97% | 98% | 100% | 100% | 100% |
| PELT (reszty STL) [model: sinus dopasowany globalnie] | 100% | 100% | 100% | 86% | 98% | 94% | 100% | 100% | 100% |

| metoda | A-L50 | A-L100 | A-L200 | B-L50 | B-L100 | B-L200 | C-L50 | C-L100 | C-L200 |
|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA [STL, okno sezonowości 7 cykli] | 0,95 | 0,97 | 0,84 | 0,90 | 0,88 | 0,66 | 0,91 | 0,95 | 0,99 |
| STL + EWMA | 0,97 | 0,98 | 0,99 | 0,93 | 0,95 | 0,96 | 0,93 | 0,96 | 0,98 |
| STL + EWMA [model: sinus dopasowany globalnie] | 0,96 | 0,98 | 0,99 | 0,93 | 0,94 | 0,95 | 0,91 | 0,95 | 0,97 |
| PELT (reszty STL) [STL, okno sezonowości 7 cykli] | 0,79 | 0,75 | 0,72 | 0,51 | 0,51 | 0,47 | 0,84 | 0,91 | 0,97 |
| PELT (reszty STL) | 0,93 | 0,95 | 0,96 | 0,78 | 0,88 | 0,89 | 0,87 | 0,90 | 0,92 |
| PELT (reszty STL) [model: sinus dopasowany globalnie] | 0,92 | 0,94 | 0,96 | 0,82 | 0,89 | 0,87 | 0,87 | 0,91 | 0,91 |

| metoda | L50 | L100 | L200 |
|---|---|---|---|
| STL + EWMA [STL, okno sezonowości 7 cykli] | 84% | 77% | 53% |
| STL + EWMA | 89% | 96% | 96% |
| STL + EWMA [model: sinus dopasowany globalnie] | 94% | 95% | 93% |
| PELT (reszty STL) [STL, okno sezonowości 7 cykli] | 59% | 55% | 50% |
| PELT (reszty STL) | 92% | 99% | 99% |
| PELT (reszty STL) [model: sinus dopasowany globalnie] | 95% | 99% | 98% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| STL + EWMA [STL, okno sezonowości 7 cykli] | 66% | 99% | 15% | 49% | 100% | 100% |
| STL + EWMA | 99% | 100% | 62% | 99% | 100% | 100% |
| STL + EWMA [model: sinus dopasowany globalnie] | 100% | 100% | 64% | 99% | 100% | 100% |
| PELT (reszty STL) [STL, okno sezonowości 7 cykli] | 16% | 98% | 0% | 13% | 100% | 100% |
| PELT (reszty STL) | 100% | 100% | 81% | 100% | 100% | 100% |
| PELT (reszty STL) [model: sinus dopasowany globalnie] | 100% | 100% | 85% | 100% | 100% | 100% |

## Analiza dodatkowa (b) Odwrotny kierunek anomalii, L = 100: poziom 2 = faza −π/4, amplituda ×1,5, okres ×1,25; poziom 3 = faza −π/2, amplituda ×2, okres ×1,5

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| STL + EWMA | 100% | 100% | 81% | 100% | 100% | 100% |
| PELT (reszty STL) | 100% | 100% | 94% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 44% | 100% | 1% | 1% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 100% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 49% | 0% | 0% | 100% | 100% |
| Isolation Forest (3 cechy) | 0% | 42% | 0% | 0% | 100% | 100% |
| autoenkoder | 60% | 100% | 91% | 100% | 100% | 100% |
| Matrix Profile bez normalizacji | 49% | 100% | 92% | 100% | 100% | 100% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| STL + EWMA | 0,97 | 0,99 | 0,93 | 0,99 | 0,93 | 0,96 |
| PELT (reszty STL) | 0,94 | 0,95 | 0,83 | 0,96 | 0,86 | 0,89 |
| Matrix Profile z normalizacją | 0,63 | 0,73 | 0,09 | 0,05 | 0,82 | 0,97 |
| cechy cykli + kNN (5 cech) | 1,00 | 1,00 | 1,00 | 1,00 | 0,99 | 1,00 |
| cechy cykli + kNN (3 cechy) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 1,00 | 1,00 | 0,99 | 1,00 | 0,99 | 1,00 |
| Isolation Forest (3 cechy) | 1,00 | 1,00 | 1,00 | 1,00 | 0,99 | 1,00 |
| autoenkoder | 0,67 | 0,75 | 0,78 | 0,88 | 0,92 | 0,99 |
| Matrix Profile bez normalizacji | 0,65 | 0,73 | 0,97 | 1,00 | 0,88 | 0,98 |

Dla porównania – kierunek z części głównej (faza +, osłabienie, przyspieszenie), poziomy 2–3, L = 100:

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| STL + EWMA | 100% | 100% | 75% | 100% | 100% | 100% |
| PELT (reszty STL) | 100% | 100% | 92% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 44% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (5 cech) | 100% | 100% | 100% | 100% | 100% | 100% |
| cechy cykli + kNN (3 cechy) | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 4% | 0% | 23% | 100% | 100% |
| Isolation Forest (3 cechy) | 0% | 12% | 0% | 33% | 100% | 100% |
| autoenkoder | 66% | 100% | 65% | 95% | 100% | 100% |
| Matrix Profile bez normalizacji | 52% | 100% | 78% | 100% | 100% | 100% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 92% | 100% | 96% | 100% | 100% | 100% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 74% | 100% | 94% | 100% | 100% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 63% | 100% | 20% | 74% | 100% | 100% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| STL + EWMA | 0,98 | 0,99 | 0,93 | 0,98 | 0,96 | 0,98 |
| PELT (reszty STL) | 0,94 | 0,96 | 0,82 | 0,95 | 0,92 | 0,88 |
| Matrix Profile z normalizacją | 0,63 | 0,71 | 0,99 | 1,00 | 0,96 | 0,97 |
| cechy cykli + kNN (5 cech) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| cechy cykli + kNN (3 cechy) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (5 cech) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| autoenkoder | 0,67 | 0,73 | 0,73 | 0,77 | 1,00 | 1,00 |
| Matrix Profile bez normalizacji | 0,66 | 0,71 | 0,95 | 0,96 | 0,96 | 0,97 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,99 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,67 | 0,72 | 0,62 | 0,72 | 1,00 | 1,00 |

## Analiza dodatkowa (c) Autoenkoder – sposób uczenia (poziomy 2–3, L = 100)

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| autoenkoder (bez odrzucania okien) | 70% | 100% | 30% | 72% | 100% | 100% |
| autoenkoder (odrzucanie 10% okien) | 70% | 100% | 57% | 90% | 100% | 100% |
| autoenkoder (uczony na szeregach bez anomalii) | 66% | 100% | 17% | 61% | 100% | 100% |

| metoda | A2 | A3 | B2 | B3 | C2 | C3 |
|---|---|---|---|---|---|---|
| autoenkoder (bez odrzucania okien) | 0,65 | 0,73 | 0,72 | 0,75 | 1,00 | 1,00 |
| autoenkoder (odrzucanie 10% okien) | 0,65 | 0,73 | 0,75 | 0,77 | 1,00 | 1,00 |
| autoenkoder (uczony na szeregach bez anomalii) | 0,65 | 0,73 | 0,62 | 0,72 | 1,00 | 1,00 |

Fałszywe alarmy na walidacji w analizach dodatkowych (odsetek szeregów z ≥1 alarmem):

|                                                                            |   odsetek |
|:---------------------------------------------------------------------------|----------:|
| ('a_model_cyklu', 'PELT (reszty STL)')                                     |     0.065 |
| ('a_model_cyklu', 'PELT (reszty STL) [STL, okno sezonowości 7 cykli]')     |     0.135 |
| ('a_model_cyklu', 'PELT (reszty STL) [model: sinus dopasowany globalnie]') |     0.085 |
| ('a_model_cyklu', 'STL + EWMA')                                            |     0.135 |
| ('a_model_cyklu', 'STL + EWMA [STL, okno sezonowości 7 cykli]')            |     0.125 |
| ('a_model_cyklu', 'STL + EWMA [model: sinus dopasowany globalnie]')        |     0.12  |
| ('b_odwrotny_kierunek', 'Isolation Forest (3 cechy)')                      |     0.12  |
| ('b_odwrotny_kierunek', 'Isolation Forest (5 cech)')                       |     0.13  |
| ('b_odwrotny_kierunek', 'Matrix Profile bez normalizacji')                 |     0.135 |
| ('b_odwrotny_kierunek', 'Matrix Profile z normalizacją')                   |     0.135 |
| ('b_odwrotny_kierunek', 'PELT (reszty STL)')                               |     0.065 |
| ('b_odwrotny_kierunek', 'STL + EWMA')                                      |     0.135 |
| ('b_odwrotny_kierunek', 'autoenkoder')                                     |     0.165 |
| ('b_odwrotny_kierunek', 'cechy cykli + kNN (3 cechy)')                     |     0.13  |
| ('b_odwrotny_kierunek', 'cechy cykli + kNN (5 cech)')                      |     0.155 |
| ('c_autoenkoder', 'autoenkoder (bez odrzucania okien)')                    |     0.13  |
| ('c_autoenkoder', 'autoenkoder (odrzucanie 10% okien)')                    |     0.165 |
| ('c_autoenkoder', 'autoenkoder (uczony na szeregach bez anomalii)')        |     0.13  |

Kary PELT w analizach dodatkowych:

| analiza             | metoda                                                |   kara_pelt |
|:--------------------|:------------------------------------------------------|------------:|
| a_model_cyklu       | PELT (reszty STL) [STL, okno sezonowości 7 cykli]     |          47 |
| a_model_cyklu       | PELT (reszty STL)                                     |          13 |
| a_model_cyklu       | PELT (reszty STL) [model: sinus dopasowany globalnie] |          14 |
| b_odwrotny_kierunek | PELT (reszty STL)                                     |          13 |
