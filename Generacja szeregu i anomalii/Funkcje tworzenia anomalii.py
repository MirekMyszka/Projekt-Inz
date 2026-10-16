
def anomaly_point(series,n,a_max,a_min, seed=50):
    rng = np.random.default_rng(seed)
    series = np.array(series, dtype=float).copy()

    # Losowe indeksy anomalii
    indices = rng.choice(
        len(series),
        size=min(n, len(series)),
        replace=False
    )

    # Losowe wartości anomalii
    anomalies = rng.uniform(a_min, a_max, size=len(indices))

    # Dodanie anomalii do szeregu
    series[indices] += anomalies

    return series