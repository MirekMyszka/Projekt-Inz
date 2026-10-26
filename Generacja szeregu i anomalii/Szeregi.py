import numpy as np

def white_noise(n=1000, seed=2137):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0, scale=1, size=n)

def AR1(n=1000, phi=0.8, seed=2137):
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=1, size=n)
    series = np.zeros(n)

    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]

    return series

def white_noise_sinus(n=1000, period=50, noise_scale=0.5, seed=2137):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    sinus = np.sin(2 * np.pi * t / period)
    noise = rng.normal(loc=0, scale=noise_scale, size=n)

    return sinus + noise