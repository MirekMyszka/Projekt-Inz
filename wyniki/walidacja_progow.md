# Sprawdzenie progów alarmu na 1 000 nowych szeregach bez anomalii

Generowane przez `src/ewaluacja/walidacja_progow.py`. Cel kalibracji: fałszywy alarm w 10% szeregów bez anomalii. Przedział ufności 95% (Wilsona).

| scenariusz | metoda | szeregi z ≥ 1 fałszywym alarmem | 95% CI | fałszywe alarmy na szereg |
|---|---|---|---|---|
| S1 | CUSUM (poziom, zmienność, zależność) | 9,8% | 8,1% – 11,8% | 0,102 |
| S1 | PELT | 8,6% | 7,0% – 10,5% | 0,086 |
| S1 | Matrix Profile | 12,4% | 10,5% – 14,6% | 0,188 |
| S1 | cechy okien + kNN (5 cech) | 9,7% | 8,0% – 11,7% | 0,115 |
| S1 | cechy okien + kNN (3 cechy) | 11,1% | 9,3% – 13,2% | 0,128 |
| S1 | Isolation Forest (5 cech) | 9,6% | 7,9% – 11,6% | 0,108 |
| S1 | Isolation Forest (3 cechy) | 5,7% | 4,4% – 7,3% | 0,057 |
| S1 | LOF (5 cech) | 11,6% | 9,8% – 13,7% | 0,130 |
| S1 | LOF (3 cechy) | 13,2% | 11,2% – 15,4% | 0,152 |
| S1 | Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 11,1% | 9,3% – 13,2% | 0,133 |
| S1 | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 7,8% | 6,3% – 9,6% | 0,087 |
| S1 | LOF (5 cech, uczony na szeregach bez anomalii) | 15,7% | 13,6% – 18,1% | 0,198 |
| S1 | LOF (3 cechy, uczony na szeregach bez anomalii) | 13,7% | 11,7% – 16,0% | 0,176 |
| S1 | Matrix Profile z normalizacją | 8,2% | 6,7% – 10,1% | 0,086 |
| S2 | reszty AR + CUSUM | 8,6% | 7,0% – 10,5% | 0,089 |
| S2 | PELT (koszt AR) | 4,0% | 3,0% – 5,4% | 0,041 |
| S2 | Matrix Profile (reszty AR) | 12,0% | 10,1% – 14,2% | 0,192 |
| S2 | cechy okien + kNN (5 cech) | 9,3% | 7,7% – 11,3% | 0,101 |
| S2 | cechy okien + kNN (3 cechy) | 10,8% | 9,0% – 12,9% | 0,117 |
| S2 | Isolation Forest (5 cech) | 10,0% | 8,3% – 12,0% | 0,107 |
| S2 | Isolation Forest (3 cechy) | 9,4% | 7,7% – 11,4% | 0,100 |
| S2 | One-Class SVM (5 cech) | 11,8% | 9,9% – 13,9% | 0,127 |
| S2 | One-Class SVM (3 cechy) | 12,3% | 10,4% – 14,5% | 0,136 |
| S2 | Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 9,9% | 8,2% – 11,9% | 0,111 |
| S2 | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 7,8% | 6,3% – 9,6% | 0,092 |
| S2 | One-Class SVM (5 cech, uczony na szeregach bez anomalii) | 12,2% | 10,3% – 14,4% | 0,151 |
| S2 | One-Class SVM (3 cechy, uczony na szeregach bez anomalii) | 12,2% | 10,3% – 14,4% | 0,139 |
| S3 | STL + EWMA | 12,1% | 10,2% – 14,3% | 0,147 |
| S3 | PELT (reszty STL) | 7,2% | 5,8% – 9,0% | 0,074 |
| S3 | Matrix Profile z normalizacją | 10,4% | 8,7% – 12,4% | 0,149 |
| S3 | cechy cykli + kNN (5 cech) | 13,1% | 11,1% – 15,3% | 0,152 |
| S3 | cechy cykli + kNN (3 cechy) | 11,6% | 9,8% – 13,7% | 0,136 |
| S3 | Isolation Forest (5 cech) | 10,5% | 8,7% – 12,6% | 0,115 |
| S3 | Isolation Forest (3 cechy) | 9,7% | 8,0% – 11,7% | 0,101 |
| S3 | autoenkoder | 10,7% | 8,9% – 12,8% | 0,373 |
| S3 | Isolation Forest (5 cech, uczony na szeregach bez anomalii) | 9,3% | 7,7% – 11,3% | 0,126 |
| S3 | Isolation Forest (3 cechy, uczony na szeregach bez anomalii) | 9,7% | 8,0% – 11,7% | 0,108 |
| S3 | autoenkoder (uczony na szeregach bez anomalii) | 9,2% | 7,6% – 11,2% | 0,100 |
| S3 | Matrix Profile bez normalizacji | 9,0% | 7,4% – 10,9% | 0,126 |
