"""Nazwy i styl wariantów metod na wykresach i w tabelach (wspólne dla wykresy_s1/s2/s3).

Metody na cechach okien występują w wariantach „(5 cech)” i „(3 cechy)”, a metody ML dodatkowo w wariancie
„uczony na szeregach bez anomalii” (ustalenia z konsultacji 27.10.2026). Kolor = rodzina metody (nazwa bazowa),
linia ciągła = 5 cech / metoda bez cech, przerywana = 3 cechy, kropkowana = uczony na szeregach bez anomalii.
"""
import re

UCZONY = "uczony na szeregach bez anomalii"
_SUFIKS = re.compile(r" \((?:5 cech|3 cechy)(?:, uczony na szeregach bez anomalii)?\)$")


def baza(nazwa: str) -> str:
    """Nazwa rodziny metody: bez dopisku o liczbie cech i sposobie uczenia."""
    if nazwa.endswith(f" ({UCZONY})"):
        return nazwa[: -len(UCZONY) - 3]
    return _SUFIKS.sub("", nazwa)


def liczba_cech(nazwa: str):
    if "(3 cechy" in nazwa:
        return 3
    if "(5 cech" in nazwa:
        return 5
    return None


def uczony(nazwa: str) -> bool:
    return UCZONY in nazwa


def rozwin(bazowe: list[str], z_cechami: set[str], ml: set[str], ae: str | None = None) -> list[str]:
    """Pełna kolejność wierszy tabel: metody główne (warianty 5 i 3 cech obok siebie), potem warianty uczone na
    szeregach bez anomalii. z_cechami - rodziny z wariantami cech, ml - rodziny z wariantem uczonym."""
    out = []
    for b in bazowe:
        out += [f"{b} (5 cech)", f"{b} (3 cechy)"] if b in z_cechami else [b]
    for b in bazowe:
        if b in ml:
            out += [f"{b} (5 cech, {UCZONY})", f"{b} (3 cechy, {UCZONY})"]
    if ae:
        out.append(f"{ae} ({UCZONY})")
    return out


def styl(nazwa: str, kolory: dict, znaczniki: dict, porownawcze=()) -> dict:
    b = baza(nazwa)
    ls = ":" if uczony(nazwa) else "--" if (liczba_cech(nazwa) == 3 or nazwa in porownawcze) else "-"
    return {"color": kolory[b], "marker": znaczniki[b], "ls": ls,
            "lw": 1.4 if nazwa in porownawcze else 1.9,
            "mfc": "white" if liczba_cech(nazwa) == 3 else kolory[b]}
