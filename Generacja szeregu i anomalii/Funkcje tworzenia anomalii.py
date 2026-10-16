
def anomaly_point(series,n,a_max,a_min, seed=50):
    rng = np.random.default_rng(seed)
    series = np.array(series, dtype=float).copy()

    indices = rng.choice(
        len(series),
        size=min(n, len(series)),
        replace=False
    )

    anomalies = rng.uniform(a_min, a_max, size=len(indices))

    series[indices] += anomalies

    return series
