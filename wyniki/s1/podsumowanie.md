# S1 – tabele wyników (generowane automatycznie przez src/ewaluacja/wykresy_s1.py)

Część główna: 1200 szeregów z anomaliami (3600 anomalii na metodę), 200 szeregów walidacyjnych bez anomalii. Próg: fałszywy alarm w 10% szeregów kalibracyjnych bez anomalii.

## Kalibracja

| metoda                                                       |     prog |   kara_pelt | skale_cusum            |
|:-------------------------------------------------------------|---------:|------------:|:-----------------------|
| CUSUM (poziom, zmienność, zależność)                         | 1.58437  |         nan | [7.351, 16.534, 16.25] |
| PELT                                                         | 0        |          62 | nan                    |
| Matrix Profile                                               | 8.88511  |         nan | nan                    |
| Matrix Profile z normalizacją                                | 7.86084  |         nan | nan                    |
| cechy okien + kNN (5 cech)                                   | 5.09588  |         nan | nan                    |
| cechy okien + kNN (3 cechy)                                  | 2.38914  |         nan | nan                    |
| Isolation Forest (5 cech)                                    | 0.670893 |         nan | nan                    |
| Isolation Forest (3 cechy)                                   | 0.687792 |         nan | nan                    |
| LOF (5 cech)                                                 | 2.79707  |         nan | nan                    |
| LOF (3 cechy)                                                | 2.89416  |         nan | nan                    |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii)  | 0.655254 |         nan | nan                    |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0.678821 |         nan | nan                    |
| LOF (5 cech, uczony na szeregach bez anomalii)               | 2.03025  |         nan | nan                    |
| LOF (3 cechy, uczony na szeregach bez anomalii)              | 2.30361  |         nan | nan                    |

## Fałszywe alarmy (walidacja na nowych szeregach bez anomalii) i miary punktowe

| metoda                                                       |   odsetek_z_fa |   fa_na_1000_pkt |   fa_w_szeregach_z_anomaliami |   precyzja |   czulosc_pkt |    f1 |
|:-------------------------------------------------------------|---------------:|-----------------:|------------------------------:|-----------:|--------------:|------:|
| CUSUM (poziom, zmienność, zależność)                         |          0.1   |            0.037 |                         0.069 |      0.891 |         0.623 | 0.653 |
| PELT                                                         |          0.095 |            0.033 |                         0.04  |      0.876 |         0.524 | 0.584 |
| Matrix Profile                                               |          0.13  |            0.088 |                         0.215 |      0.975 |         0.433 | 0.56  |
| cechy okien + kNN (5 cech)                                   |          0.085 |            0.03  |                         0.092 |      0.957 |         0.385 | 0.483 |
| cechy okien + kNN (3 cechy)                                  |          0.125 |            0.043 |                         0.052 |      0.987 |         0.555 | 0.65  |
| Isolation Forest (5 cech)                                    |          0.105 |            0.048 |                         0.014 |      0.983 |         0.186 | 0.269 |
| Isolation Forest (3 cechy)                                   |          0.04  |            0.013 |                         0.009 |      0.987 |         0.128 | 0.19  |
| LOF (5 cech)                                                 |          0.1   |            0.037 |                         0.108 |      0.953 |         0.299 | 0.399 |
| LOF (3 cechy)                                                |          0.165 |            0.065 |                         0.092 |      0.963 |         0.333 | 0.442 |
| Matrix Profile z normalizacją                                |          0.085 |            0.035 |                         0.079 |      0.077 |         0     | 0     |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii)  |          0.105 |            0.043 |                         0.047 |      0.968 |         0.262 | 0.357 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) |          0.075 |            0.025 |                         0.022 |      0.982 |         0.237 | 0.329 |
| LOF (5 cech, uczony na szeregach bez anomalii)               |          0.14  |            0.06  |                         0.072 |      0.978 |         0.532 | 0.627 |
| LOF (3 cechy, uczony na szeregach bez anomalii)              |          0.105 |            0.048 |                         0.057 |      0.981 |         0.54  | 0.637 |

Poziomy: A Δ = 0,5 / 1 / 2 / 3 σ; B c = 1,5 / 2 / 3 / 4; C φ = 0,3 / 0,5 / 0,7 / 0,9.

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = 20

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 1% | 54% | 100% | 100% | 20% | 66% | 97% | 100% | 1% | 4% | 10% | 22% |
| PELT | 2% | 12% | 100% | 100% | 22% | 74% | 98% | 100% | 1% | 3% | 5% | 11% |
| Matrix Profile | 1% | 13% | 99% | 100% | 35% | 78% | 100% | 100% | 0% | 1% | 2% | 6% |
| cechy okien + kNN (5 cech) | 0% | 0% | 79% | 100% | 9% | 41% | 96% | 99% | 0% | 0% | 2% | 0% |
| cechy okien + kNN (3 cechy) | 2% | 34% | 100% | 100% | 25% | 71% | 98% | 100% | 1% | 1% | 9% | 24% |
| Isolation Forest (5 cech) | 1% | 2% | 54% | 99% | 15% | 54% | 95% | 98% | 1% | 0% | 0% | 0% |
| Isolation Forest (3 cechy) | 0% | 7% | 96% | 100% | 3% | 21% | 44% | 69% | 1% | 0% | 0% | 0% |
| LOF (5 cech) | 1% | 10% | 94% | 97% | 7% | 41% | 89% | 97% | 1% | 0% | 1% | 13% |
| LOF (3 cechy) | 2% | 21% | 91% | 92% | 19% | 56% | 90% | 93% | 1% | 2% | 7% | 20% |
| Matrix Profile z normalizacją | 1% | 0% | 0% | 0% | 0% | 0% | 1% | 0% | 0% | 0% | 0% | 1% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0% | 9% | 95% | 100% | 18% | 52% | 97% | 99% | 0% | 0% | 2% | 7% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 1% | 21% | 100% | 100% | 8% | 21% | 36% | 73% | 1% | 3% | 6% | 20% |
| LOF (5 cech, uczony na szeregach bez anomalii) | 4% | 33% | 100% | 100% | 13% | 56% | 98% | 100% | 1% | 0% | 5% | 19% |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 4% | 41% | 100% | 100% | 17% | 58% | 97% | 100% | 1% | 1% | 6% | 16% |

## AUC – rodzaj × poziom intensywności, L = 20

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,60 | 0,91 | 1,00 | 1,00 | 0,69 | 0,87 | 0,97 | 0,98 | 0,63 | 0,73 | 0,88 | 0,94 |
| PELT | 0,50 | 0,55 | 0,98 | 0,99 | 0,58 | 0,81 | 0,96 | 0,98 | 0,50 | 0,50 | 0,51 | 0,54 |
| Matrix Profile | 0,63 | 0,89 | 1,00 | 1,00 | 0,88 | 0,99 | 1,00 | 1,00 | 0,52 | 0,50 | 0,44 | 0,48 |
| cechy okien + kNN (5 cech) | 0,69 | 0,94 | 1,00 | 1,00 | 0,82 | 0,98 | 1,00 | 1,00 | 0,56 | 0,68 | 0,78 | 0,87 |
| cechy okien + kNN (3 cechy) | 0,69 | 0,96 | 1,00 | 1,00 | 0,83 | 0,98 | 1,00 | 1,00 | 0,60 | 0,70 | 0,81 | 0,88 |
| Isolation Forest (5 cech) | 0,68 | 0,94 | 1,00 | 1,00 | 0,83 | 0,98 | 1,00 | 1,00 | 0,56 | 0,69 | 0,80 | 0,88 |
| Isolation Forest (3 cechy) | 0,70 | 0,97 | 1,00 | 1,00 | 0,83 | 0,98 | 1,00 | 1,00 | 0,61 | 0,71 | 0,82 | 0,88 |
| LOF (5 cech) | 0,65 | 0,90 | 0,99 | 1,00 | 0,76 | 0,93 | 0,99 | 0,99 | 0,55 | 0,67 | 0,73 | 0,85 |
| LOF (3 cechy) | 0,65 | 0,92 | 0,99 | 0,97 | 0,79 | 0,96 | 0,98 | 0,96 | 0,58 | 0,66 | 0,77 | 0,85 |
| Matrix Profile z normalizacją | 0,50 | 0,50 | 0,47 | 0,44 | 0,47 | 0,51 | 0,52 | 0,47 | 0,47 | 0,51 | 0,49 | 0,42 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,69 | 0,93 | 1,00 | 1,00 | 0,84 | 0,98 | 1,00 | 1,00 | 0,55 | 0,68 | 0,80 | 0,87 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,71 | 0,98 | 1,00 | 1,00 | 0,84 | 0,97 | 0,99 | 1,00 | 0,60 | 0,71 | 0,82 | 0,88 |
| LOF (5 cech, uczony na szeregach bez anomalii) | 0,67 | 0,95 | 1,00 | 1,00 | 0,77 | 0,95 | 1,00 | 1,00 | 0,54 | 0,67 | 0,78 | 0,87 |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 0,69 | 0,97 | 1,00 | 1,00 | 0,80 | 0,96 | 1,00 | 1,00 | 0,58 | 0,69 | 0,77 | 0,86 |

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = 50

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 17% | 98% | 100% | 100% | 41% | 99% | 100% | 100% | 1% | 12% | 43% | 93% |
| PELT | 6% | 60% | 100% | 100% | 60% | 99% | 100% | 100% | 4% | 1% | 14% | 42% |
| Matrix Profile | 5% | 70% | 100% | 100% | 77% | 99% | 100% | 100% | 1% | 1% | 6% | 14% |
| cechy okien + kNN (5 cech) | 0% | 49% | 100% | 100% | 27% | 96% | 100% | 100% | 0% | 0% | 2% | 54% |
| cechy okien + kNN (3 cechy) | 32% | 98% | 100% | 100% | 60% | 99% | 100% | 100% | 7% | 23% | 76% | 99% |
| Isolation Forest (5 cech) | 1% | 5% | 35% | 79% | 37% | 90% | 100% | 100% | 1% | 0% | 0% | 14% |
| Isolation Forest (3 cechy) | 3% | 20% | 89% | 100% | 22% | 49% | 82% | 86% | 2% | 1% | 4% | 27% |
| LOF (5 cech) | 9% | 82% | 99% | 100% | 39% | 97% | 100% | 97% | 1% | 5% | 30% | 94% |
| LOF (3 cechy) | 25% | 94% | 99% | 100% | 54% | 98% | 99% | 99% | 7% | 15% | 57% | 97% |
| Matrix Profile z normalizacją | 0% | 0% | 1% | 0% | 1% | 1% | 1% | 0% | 0% | 0% | 0% | 0% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 4% | 16% | 99% | 100% | 35% | 83% | 100% | 100% | 2% | 2% | 15% | 72% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 5% | 43% | 100% | 100% | 20% | 51% | 82% | 91% | 6% | 11% | 39% | 83% |
| LOF (5 cech, uczony na szeregach bez anomalii) | 36% | 99% | 100% | 100% | 43% | 99% | 100% | 100% | 7% | 10% | 60% | 97% |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 39% | 98% | 100% | 100% | 54% | 99% | 100% | 100% | 3% | 15% | 55% | 99% |

## AUC – rodzaj × poziom intensywności, L = 50

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,69 | 0,97 | 1,00 | 1,00 | 0,76 | 0,95 | 0,99 | 0,99 | 0,71 | 0,91 | 0,98 | 0,99 |
| PELT | 0,51 | 0,76 | 0,99 | 1,00 | 0,74 | 0,95 | 0,98 | 0,99 | 0,51 | 0,49 | 0,52 | 0,62 |
| Matrix Profile | 0,71 | 0,95 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 | 0,55 | 0,44 | 0,46 | 0,38 |
| cechy okien + kNN (5 cech) | 0,85 | 0,99 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,70 | 0,83 | 0,94 | 0,98 |
| cechy okien + kNN (3 cechy) | 0,88 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,73 | 0,88 | 0,96 | 0,99 |
| Isolation Forest (5 cech) | 0,83 | 0,98 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,72 | 0,86 | 0,96 | 0,99 |
| Isolation Forest (3 cechy) | 0,89 | 1,00 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,75 | 0,90 | 0,96 | 0,99 |
| LOF (5 cech) | 0,83 | 0,99 | 1,00 | 1,00 | 0,89 | 0,99 | 1,00 | 1,00 | 0,68 | 0,83 | 0,94 | 0,98 |
| LOF (3 cechy) | 0,85 | 0,99 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 0,99 | 0,71 | 0,85 | 0,94 | 0,98 |
| Matrix Profile z normalizacją | 0,50 | 0,49 | 0,42 | 0,28 | 0,52 | 0,52 | 0,50 | 0,50 | 0,50 | 0,48 | 0,44 | 0,26 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,83 | 0,95 | 0,99 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,71 | 0,84 | 0,95 | 0,98 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,90 | 0,99 | 1,00 | 1,00 | 0,94 | 0,99 | 1,00 | 1,00 | 0,74 | 0,89 | 0,96 | 0,99 |
| LOF (5 cech, uczony na szeregach bez anomalii) | 0,86 | 0,99 | 1,00 | 1,00 | 0,91 | 0,99 | 1,00 | 1,00 | 0,67 | 0,83 | 0,95 | 0,98 |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 0,89 | 1,00 | 1,00 | 1,00 | 0,91 | 1,00 | 1,00 | 1,00 | 0,70 | 0,86 | 0,96 | 0,98 |

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = 100

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 38% | 100% | 100% | 100% | 88% | 100% | 100% | 100% | 19% | 67% | 100% | 100% |
| PELT | 12% | 99% | 100% | 100% | 97% | 100% | 100% | 100% | 8% | 18% | 41% | 82% |
| Matrix Profile | 12% | 68% | 99% | 100% | 99% | 100% | 100% | 100% | 2% | 6% | 13% | 17% |
| cechy okien + kNN (5 cech) | 1% | 56% | 100% | 100% | 40% | 98% | 100% | 100% | 1% | 0% | 5% | 67% |
| cechy okien + kNN (3 cechy) | 53% | 100% | 100% | 100% | 90% | 100% | 100% | 100% | 12% | 55% | 98% | 100% |
| Isolation Forest (5 cech) | 1% | 0% | 5% | 24% | 43% | 73% | 100% | 100% | 1% | 0% | 0% | 1% |
| Isolation Forest (3 cechy) | 0% | 3% | 19% | 39% | 14% | 34% | 56% | 57% | 0% | 0% | 0% | 5% |
| LOF (5 cech) | 13% | 73% | 86% | 87% | 51% | 72% | 84% | 91% | 4% | 13% | 59% | 95% |
| LOF (3 cechy) | 30% | 89% | 93% | 81% | 71% | 86% | 90% | 83% | 10% | 33% | 78% | 99% |
| Matrix Profile z normalizacją | 1% | 1% | 1% | 1% | 0% | 0% | 0% | 1% | 2% | 0% | 1% | 0% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 4% | 12% | 96% | 100% | 53% | 92% | 100% | 100% | 1% | 10% | 27% | 93% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 9% | 33% | 100% | 100% | 39% | 64% | 93% | 96% | 6% | 35% | 70% | 99% |
| LOF (5 cech, uczony na szeregach bez anomalii) | 56% | 100% | 100% | 100% | 81% | 100% | 100% | 100% | 5% | 33% | 90% | 100% |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 62% | 100% | 100% | 100% | 86% | 100% | 100% | 100% | 7% | 34% | 88% | 100% |

## AUC – rodzaj × poziom intensywności, L = 100

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,70 | 0,98 | 1,00 | 1,00 | 0,84 | 0,97 | 0,99 | 1,00 | 0,80 | 0,98 | 0,99 | 0,99 |
| PELT | 0,54 | 0,96 | 0,99 | 1,00 | 0,90 | 0,97 | 0,99 | 1,00 | 0,51 | 0,53 | 0,60 | 0,79 |
| Matrix Profile | 0,71 | 0,93 | 0,96 | 0,97 | 0,98 | 1,00 | 1,00 | 1,00 | 0,46 | 0,51 | 0,51 | 0,31 |
| cechy okien + kNN (5 cech) | 0,88 | 0,99 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 | 0,72 | 0,90 | 0,97 | 0,99 |
| cechy okien + kNN (3 cechy) | 0,91 | 1,00 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 | 0,76 | 0,93 | 0,97 | 0,99 |
| Isolation Forest (5 cech) | 0,85 | 0,98 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 | 0,74 | 0,91 | 0,97 | 0,99 |
| Isolation Forest (3 cechy) | 0,91 | 1,00 | 1,00 | 1,00 | 0,96 | 1,00 | 1,00 | 1,00 | 0,78 | 0,94 | 0,98 | 0,99 |
| LOF (5 cech) | 0,85 | 0,97 | 0,94 | 0,90 | 0,91 | 0,92 | 0,83 | 0,73 | 0,70 | 0,88 | 0,95 | 0,98 |
| LOF (3 cechy) | 0,87 | 0,97 | 0,94 | 0,92 | 0,93 | 0,96 | 0,89 | 0,84 | 0,71 | 0,88 | 0,95 | 0,98 |
| Matrix Profile z normalizacją | 0,51 | 0,49 | 0,42 | 0,32 | 0,50 | 0,49 | 0,50 | 0,51 | 0,48 | 0,45 | 0,35 | 0,14 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,85 | 0,94 | 0,97 | 0,97 | 0,97 | 1,00 | 1,00 | 1,00 | 0,73 | 0,91 | 0,96 | 0,99 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,92 | 0,99 | 1,00 | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 | 0,78 | 0,94 | 0,98 | 0,99 |
| LOF (5 cech, uczony na szeregach bez anomalii) | 0,89 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 | 0,71 | 0,90 | 0,97 | 0,99 |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 0,91 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 | 0,73 | 0,92 | 0,97 | 0,99 |

## Odsetek wykrytych anomalii – rodzaj × poziom intensywności (średnio po L)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 19% | 84% | 100% | 100% | 50% | 88% | 99% | 100% | 7% | 28% | 51% | 72% |
| PELT | 7% | 57% | 100% | 100% | 60% | 91% | 99% | 100% | 4% | 7% | 20% | 45% |
| Matrix Profile | 6% | 50% | 99% | 100% | 70% | 92% | 100% | 100% | 1% | 3% | 7% | 12% |
| cechy okien + kNN (5 cech) | 0% | 35% | 93% | 100% | 25% | 78% | 99% | 100% | 0% | 0% | 3% | 40% |
| cechy okien + kNN (3 cechy) | 29% | 77% | 100% | 100% | 58% | 90% | 99% | 100% | 7% | 26% | 61% | 74% |
| Isolation Forest (5 cech) | 1% | 2% | 31% | 67% | 32% | 72% | 98% | 99% | 1% | 0% | 0% | 5% |
| Isolation Forest (3 cechy) | 1% | 10% | 68% | 80% | 13% | 35% | 61% | 71% | 1% | 0% | 1% | 11% |
| LOF (5 cech) | 8% | 55% | 93% | 95% | 32% | 70% | 91% | 95% | 2% | 6% | 30% | 67% |
| LOF (3 cechy) | 19% | 68% | 94% | 91% | 48% | 80% | 93% | 92% | 6% | 17% | 47% | 72% |
| Matrix Profile z normalizacją | 1% | 0% | 1% | 0% | 0% | 0% | 1% | 0% | 1% | 0% | 0% | 0% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 3% | 12% | 97% | 100% | 35% | 76% | 99% | 100% | 1% | 4% | 15% | 57% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 5% | 32% | 100% | 100% | 22% | 45% | 70% | 87% | 4% | 16% | 38% | 67% |
| LOF (5 cech, uczony na szeregach bez anomalii) | 32% | 77% | 100% | 100% | 46% | 85% | 99% | 100% | 4% | 14% | 52% | 72% |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 35% | 80% | 100% | 100% | 52% | 86% | 99% | 100% | 4% | 17% | 50% | 72% |

## AUC – rodzaj × poziom intensywności (średnio po L)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,66 | 0,95 | 1,00 | 1,00 | 0,76 | 0,93 | 0,98 | 0,99 | 0,71 | 0,87 | 0,95 | 0,98 |
| PELT | 0,52 | 0,75 | 0,99 | 1,00 | 0,74 | 0,91 | 0,98 | 0,99 | 0,50 | 0,51 | 0,54 | 0,65 |
| Matrix Profile | 0,69 | 0,92 | 0,99 | 0,99 | 0,95 | 1,00 | 1,00 | 1,00 | 0,51 | 0,48 | 0,47 | 0,39 |
| cechy okien + kNN (5 cech) | 0,81 | 0,97 | 1,00 | 1,00 | 0,91 | 0,99 | 1,00 | 1,00 | 0,66 | 0,80 | 0,90 | 0,95 |
| cechy okien + kNN (3 cechy) | 0,83 | 0,99 | 1,00 | 1,00 | 0,91 | 0,99 | 1,00 | 1,00 | 0,70 | 0,84 | 0,91 | 0,95 |
| Isolation Forest (5 cech) | 0,78 | 0,96 | 1,00 | 1,00 | 0,91 | 0,99 | 1,00 | 1,00 | 0,68 | 0,82 | 0,91 | 0,95 |
| Isolation Forest (3 cechy) | 0,83 | 0,99 | 1,00 | 1,00 | 0,91 | 0,99 | 1,00 | 1,00 | 0,71 | 0,85 | 0,92 | 0,95 |
| LOF (5 cech) | 0,78 | 0,95 | 0,98 | 0,97 | 0,85 | 0,95 | 0,94 | 0,91 | 0,64 | 0,79 | 0,87 | 0,93 |
| LOF (3 cechy) | 0,79 | 0,96 | 0,98 | 0,96 | 0,88 | 0,97 | 0,96 | 0,93 | 0,67 | 0,80 | 0,89 | 0,94 |
| Matrix Profile z normalizacją | 0,50 | 0,50 | 0,44 | 0,35 | 0,50 | 0,51 | 0,51 | 0,49 | 0,49 | 0,48 | 0,43 | 0,28 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,79 | 0,94 | 0,99 | 0,99 | 0,92 | 0,99 | 1,00 | 1,00 | 0,66 | 0,81 | 0,90 | 0,95 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,84 | 0,99 | 1,00 | 1,00 | 0,91 | 0,99 | 1,00 | 1,00 | 0,71 | 0,84 | 0,92 | 0,95 |
| LOF (5 cech, uczony na szeregach bez anomalii) | 0,81 | 0,98 | 1,00 | 1,00 | 0,87 | 0,98 | 1,00 | 1,00 | 0,64 | 0,80 | 0,90 | 0,95 |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 0,83 | 0,99 | 1,00 | 1,00 | 0,89 | 0,98 | 1,00 | 1,00 | 0,67 | 0,82 | 0,90 | 0,94 |

## Odsetek wykrytych – rodzaj × długość L (średnio po poziomach)

| metoda | A-L20 | A-L50 | A-L100 | B-L20 | B-L50 | B-L100 | C-L20 | C-L50 | C-L100 |
|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 64% | 79% | 84% | 71% | 85% | 97% | 9% | 37% | 72% |
| PELT | 54% | 66% | 78% | 74% | 90% | 99% | 5% | 15% | 37% |
| Matrix Profile | 53% | 69% | 70% | 78% | 94% | 100% | 2% | 6% | 10% |
| cechy okien + kNN (5 cech) | 45% | 62% | 64% | 61% | 81% | 84% | 0% | 14% | 18% |
| cechy okien + kNN (3 cechy) | 59% | 82% | 88% | 74% | 90% | 98% | 9% | 51% | 66% |
| Isolation Forest (5 cech) | 39% | 30% | 8% | 66% | 82% | 79% | 0% | 4% | 0% |
| Isolation Forest (3 cechy) | 51% | 53% | 15% | 34% | 60% | 40% | 0% | 8% | 1% |
| LOF (5 cech) | 50% | 72% | 65% | 58% | 83% | 74% | 4% | 32% | 43% |
| LOF (3 cechy) | 52% | 80% | 73% | 64% | 88% | 82% | 8% | 44% | 55% |
| Matrix Profile z normalizacją | 0% | 0% | 1% | 0% | 1% | 0% | 0% | 0% | 1% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 51% | 55% | 53% | 66% | 80% | 86% | 2% | 23% | 33% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 56% | 62% | 60% | 34% | 61% | 73% | 8% | 35% | 52% |
| LOF (5 cech, uczony na szeregach bez anomalii) | 59% | 84% | 89% | 67% | 86% | 95% | 6% | 44% | 57% |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 61% | 84% | 90% | 68% | 88% | 96% | 6% | 43% | 57% |

## Średnio dla rodzaju (wszystkie poziomy i L)

| metoda | A | B | C |
|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 76% | 84% | 39% |
| PELT | 66% | 88% | 19% |
| Matrix Profile | 64% | 91% | 6% |
| cechy okien + kNN (5 cech) | 57% | 76% | 11% |
| cechy okien + kNN (3 cechy) | 77% | 87% | 42% |
| Isolation Forest (5 cech) | 26% | 75% | 2% |
| Isolation Forest (3 cechy) | 40% | 45% | 3% |
| LOF (5 cech) | 63% | 72% | 26% |
| LOF (3 cechy) | 68% | 78% | 36% |
| Matrix Profile z normalizacją | 0% | 0% | 0% |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 53% | 77% | 19% |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 59% | 56% | 32% |
| LOF (5 cech, uczony na szeregach bez anomalii) | 77% | 82% | 36% |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 79% | 84% | 35% |

| metoda | A | B | C |
|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,90 | 0,92 | 0,88 |
| PELT | 0,81 | 0,90 | 0,55 |
| Matrix Profile | 0,90 | 0,99 | 0,46 |
| cechy okien + kNN (5 cech) | 0,94 | 0,97 | 0,83 |
| cechy okien + kNN (3 cechy) | 0,95 | 0,98 | 0,85 |
| Isolation Forest (5 cech) | 0,94 | 0,98 | 0,84 |
| Isolation Forest (3 cechy) | 0,95 | 0,98 | 0,86 |
| LOF (5 cech) | 0,92 | 0,91 | 0,81 |
| LOF (3 cechy) | 0,92 | 0,93 | 0,82 |
| Matrix Profile z normalizacją | 0,45 | 0,50 | 0,42 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,93 | 0,98 | 0,83 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,96 | 0,97 | 0,86 |
| LOF (5 cech, uczony na szeregach bez anomalii) | 0,95 | 0,96 | 0,82 |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 0,95 | 0,97 | 0,83 |

## AP (PR-AUC) średnio dla rodzaju

| metoda | A | B | C |
|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,78 | 0,81 | 0,45 |
| PELT | 0,63 | 0,80 | 0,12 |
| Matrix Profile | 0,62 | 0,91 | 0,07 |
| cechy okien + kNN (5 cech) | 0,72 | 0,85 | 0,35 |
| cechy okien + kNN (3 cechy) | 0,79 | 0,88 | 0,46 |
| Isolation Forest (5 cech) | 0,65 | 0,86 | 0,38 |
| Isolation Forest (3 cechy) | 0,77 | 0,84 | 0,47 |
| LOF (5 cech) | 0,65 | 0,64 | 0,36 |
| LOF (3 cechy) | 0,66 | 0,67 | 0,39 |
| Matrix Profile z normalizacją | 0,02 | 0,03 | 0,02 |
| Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 0,54 | 0,85 | 0,35 |
| Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 0,72 | 0,72 | 0,45 |
| LOF (5 cech, uczony na szeregach bez anomalii) | 0,78 | 0,84 | 0,42 |
| LOF (3 cechy, uczony na szeregach bez anomalii) | 0,81 | 0,86 | 0,45 |

## Długość okna w (poziomy 2 i 3, wszystkie L)

| metoda | A-w20 | A-w50 | A-w100 | B-w20 | B-w50 | B-w100 | C-w20 | C-w50 | C-w100 |
|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile | 56% | 72% | 75% | 98% | 98% | 95% | 3% | 4% | 3% |
| cechy okien + kNN (5 cech) | 32% | 62% | 74% | 80% | 90% | 91% | 0% | 4% | 17% |
| cechy okien + kNN (3 cechy) | 83% | 88% | 82% | 97% | 95% | 90% | 28% | 46% | 39% |
| Isolation Forest (5 cech) | 9% | 17% | 12% | 93% | 88% | 82% | 0% | 0% | 0% |
| Isolation Forest (3 cechy) | 35% | 39% | 48% | 80% | 46% | 24% | 6% | 2% | 1% |
| LOF (5 cech) | 48% | 76% | 75% | 69% | 85% | 86% | 10% | 18% | 24% |
| LOF (3 cechy) | 56% | 82% | 73% | 76% | 87% | 77% | 17% | 36% | 26% |

| metoda | A-w20-L20 | A-w20-L50 | A-w20-L100 | A-w50-L20 | A-w50-L50 | A-w50-L100 | A-w100-L20 | A-w100-L50 | A-w100-L100 | B-w20-L20 | B-w20-L50 | B-w20-L100 | B-w50-L20 | B-w50-L50 | B-w50-L100 | B-w100-L20 | B-w100-L50 | B-w100-L100 | C-w20-L20 | C-w20-L50 | C-w20-L100 | C-w50-L20 | C-w50-L50 | C-w50-L100 | C-w100-L20 | C-w100-L50 | C-w100-L100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matrix Profile | 61% | 57% | 51% | 60% | 79% | 76% | 55% | 80% | 89% | 95% | 100% | 100% | 94% | 100% | 100% | 84% | 100% | 100% | 2% | 3% | 3% | 1% | 5% | 6% | 1% | 2% | 7% |
| cechy okien + kNN (5 cech) | 44% | 44% | 9% | 43% | 72% | 71% | 45% | 76% | 100% | 73% | 88% | 78% | 72% | 98% | 100% | 74% | 99% | 100% | 0% | 0% | 0% | 0% | 3% | 8% | 0% | 3% | 49% |
| cechy okien + kNN (3 cechy) | 80% | 90% | 80% | 68% | 97% | 100% | 53% | 94% | 100% | 91% | 100% | 100% | 85% | 100% | 100% | 71% | 100% | 100% | 11% | 31% | 43% | 8% | 52% | 77% | 3% | 28% | 85% |
| Isolation Forest (5 cech) | 20% | 4% | 2% | 33% | 13% | 4% | 9% | 22% | 6% | 88% | 98% | 92% | 79% | 97% | 89% | 70% | 86% | 89% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Isolation Forest (3 cechy) | 55% | 35% | 14% | 55% | 51% | 12% | 34% | 55% | 56% | 68% | 88% | 85% | 36% | 54% | 49% | 14% | 32% | 27% | 5% | 9% | 3% | 0% | 7% | 0% | 1% | 0% | 3% |
| LOF (5 cech) | 69% | 60% | 14% | 56% | 91% | 80% | 43% | 83% | 100% | 82% | 86% | 39% | 78% | 98% | 79% | 68% | 90% | 99% | 2% | 11% | 17% | 0% | 19% | 34% | 0% | 14% | 59% |
| LOF (3 cechy) | 75% | 71% | 23% | 59% | 96% | 91% | 40% | 81% | 99% | 86% | 96% | 46% | 74% | 100% | 87% | 52% | 85% | 94% | 6% | 20% | 26% | 7% | 42% | 59% | 1% | 18% | 58% |

## Poziom szumu σ (Δ = 1 bezwzględnie, c = 2, φ = 0,5, L = 50)

| metoda | A-σ0.5 | A-σ1.0 | A-σ2.0 | B-σ0.5 | B-σ1.0 | B-σ2.0 | C-σ0.5 | C-σ1.0 | C-σ2.0 |
|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 100% | 98% | 20% | 99% | 99% | 99% | 17% | 17% | 17% |
| PELT | 100% | 57% | 6% | 100% | 100% | 100% | 5% | 5% | 5% |
| Matrix Profile | 100% | 64% | 11% | 100% | 100% | 100% | 2% | 2% | 2% |
| cechy okien + kNN (5 cech) | 100% | 42% | 0% | 98% | 98% | 98% | 0% | 0% | 0% |
| cechy okien + kNN (3 cechy) | 100% | 98% | 24% | 100% | 100% | 100% | 31% | 33% | 32% |
| Isolation Forest (5 cech) | 68% | 4% | 0% | 80% | 88% | 94% | 1% | 1% | 2% |
| Isolation Forest (3 cechy) | 96% | 9% | 0% | 28% | 43% | 54% | 0% | 3% | 9% |
| LOF (5 cech) | 100% | 84% | 7% | 91% | 99% | 100% | 10% | 9% | 10% |
| LOF (3 cechy) | 100% | 94% | 17% | 99% | 100% | 100% | 27% | 28% | 27% |
| Matrix Profile z normalizacją | 0% | 0% | 1% | 0% | 0% | 0% | 0% | 0% | 0% |

| metoda | A-σ0.5 | A-σ1.0 | A-σ2.0 | B-σ0.5 | B-σ1.0 | B-σ2.0 | C-σ0.5 | C-σ1.0 | C-σ2.0 |
|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 1,00 | 0,96 | 0,64 | 0,95 | 0,95 | 0,96 | 0,92 | 0,91 | 0,91 |
| PELT | 0,99 | 0,74 | 0,51 | 0,96 | 0,95 | 0,96 | 0,51 | 0,51 | 0,51 |
| Matrix Profile | 1,00 | 0,96 | 0,72 | 1,00 | 1,00 | 1,00 | 0,51 | 0,51 | 0,51 |
| cechy okien + kNN (5 cech) | 1,00 | 0,99 | 0,82 | 1,00 | 1,00 | 1,00 | 0,87 | 0,87 | 0,87 |
| cechy okien + kNN (3 cechy) | 1,00 | 1,00 | 0,85 | 1,00 | 1,00 | 1,00 | 0,90 | 0,90 | 0,90 |
| Isolation Forest (5 cech) | 1,00 | 0,98 | 0,80 | 1,00 | 1,00 | 1,00 | 0,88 | 0,88 | 0,89 |
| Isolation Forest (3 cechy) | 1,00 | 1,00 | 0,87 | 1,00 | 1,00 | 1,00 | 0,91 | 0,91 | 0,91 |
| LOF (5 cech) | 1,00 | 0,99 | 0,81 | 1,00 | 1,00 | 1,00 | 0,84 | 0,84 | 0,84 |
| LOF (3 cechy) | 1,00 | 0,99 | 0,81 | 1,00 | 1,00 | 1,00 | 0,87 | 0,87 | 0,87 |
| Matrix Profile z normalizacją | 0,47 | 0,51 | 0,52 | 0,50 | 0,50 | 0,50 | 0,48 | 0,51 | 0,51 |

Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem) dla każdego σ:

| metoda | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 10% | 10% | 10% |
| PELT | 10% | 10% | 10% |
| Matrix Profile | 13% | 13% | 13% |
| cechy okien + kNN (5 cech) | 8% | 8% | 8% |
| cechy okien + kNN (3 cechy) | 12% | 12% | 12% |
| Isolation Forest (5 cech) | 10% | 10% | 10% |
| Isolation Forest (3 cechy) | 4% | 4% | 4% |
| LOF (5 cech) | 10% | 10% | 10% |
| LOF (3 cechy) | 16% | 16% | 16% |
| Matrix Profile z normalizacją | 8% | 8% | 8% |

## Analiza dodatkowa: metody patrzące tylko na zależność (L = 50)

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 14% | 98% | 100% | 100% | 46% | 99% | 100% | 100% | 2% | 13% | 48% | 87% |
| CUSUM: tylko karta zależności | 2% | 2% | 0% | 0% | 2% | 0% | 2% | 0% | 7% | 22% | 61% | 100% |
| cechy okien + kNN (5 cech) | 2% | 51% | 100% | 100% | 29% | 97% | 100% | 100% | 0% | 0% | 8% | 51% |
| kNN: średnia, odch. std, autokorelacja | 29% | 99% | 100% | 100% | 70% | 100% | 100% | 100% | 5% | 24% | 76% | 99% |
| kNN: tylko autokorelacja | 0% | 1% | 14% | 24% | 1% | 0% | 3% | 1% | 7% | 28% | 45% | 33% |

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM (poziom, zmienność, zależność) | 0,69 | 0,96 | 1,00 | 1,00 | 0,76 | 0,95 | 0,99 | 0,99 | 0,74 | 0,90 | 0,98 | 0,99 |
| CUSUM: tylko karta zależności | 0,52 | 0,48 | 0,50 | 0,50 | 0,33 | 0,29 | 0,26 | 0,25 | 0,72 | 0,89 | 0,97 | 0,99 |
| cechy okien + kNN (5 cech) | 0,85 | 0,99 | 1,00 | 1,00 | 0,94 | 1,00 | 1,00 | 1,00 | 0,70 | 0,86 | 0,95 | 0,98 |
| kNN: średnia, odch. std, autokorelacja | 0,89 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 1,00 | 0,74 | 0,88 | 0,96 | 0,98 |
| kNN: tylko autokorelacja | 0,55 | 0,65 | 0,89 | 0,97 | 0,48 | 0,53 | 0,52 | 0,54 | 0,75 | 0,88 | 0,95 | 0,95 |

Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem):

| metoda                                 |   odsetek |
|:---------------------------------------|----------:|
| CUSUM (poziom, zmienność, zależność)   |     0.1   |
| CUSUM: tylko karta zależności          |     0.18  |
| cechy okien + kNN (5 cech)             |     0.085 |
| kNN: tylko autokorelacja               |     0.085 |
| kNN: średnia, odch. std, autokorelacja |     0.125 |
