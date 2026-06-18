"""
nombres.py — Normalización robusta de nombres de selección
==========================================================
Resuelve los nombres que llegan de fuentes externas (ESPN, The Odds API,
ranking FIFA…) al nombre canónico del CSV de resultados, SIN tener que
enumerar cada variante. Estrategia, en orden:

  1. Coincidencia exacta con un nombre del dataset.
  2. Alias curados (sinónimos reales que no se pueden derivar:
     USA→United States, Korea Republic→South Korea, Cabo Verde→Cape Verde…).
  3. Normalización por TOKENS: minúsculas, sin acentos ni puntuación, sin
     palabras vacías, "st."→"saint", y tokens ORDENADOS. Esto captura solo
     reordenamientos/acentos/puntuación (p.ej. "Congo DR" == "DR Congo",
     "Côte d'Ivoire" == "Cote d Ivoire", "The Gambia" == "Gambia").
  4. (Opcional) coincidencia aproximada (difflib), desactivada por defecto
     para no arriesgar falsos positivos entre países parecidos.

Si nada casa, devuelve el nombre original tal cual (no rompe nada).
"""
import re
import functools
import unicodedata
from difflib import get_close_matches

CSV_DEFAULT = "results_actualizado_mundial_2026.csv"

# Sinónimos reales (palabras distintas) que la normalización NO puede derivar.
ALIAS = {
    "USA": "United States", "United States of America": "United States",
    "US Virgin Islands": "United States Virgin Islands",
    "Korea Republic": "South Korea", "Korea DPR": "North Korea",
    "DPR Korea": "North Korea",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "China": "China PR",
    "Chinese Taipei": "Taiwan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Ireland": "Republic of Ireland",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Brunei Darussalam": "Brunei",
    "Hong Kong, China": "Hong Kong",
}

_STOP = {"and", "the", "of"}
_EXPAND = {"st": "saint"}


def _norm(s):
    """Clave normalizada por tokens ordenados (sin acentos/puntuación/stopwords)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    toks = [_EXPAND.get(t, t) for t in s.split() if t and t not in _STOP]
    return " ".join(sorted(toks))


class Resolver:
    def __init__(self, canonicos, alias=None):
        self.canon = set(canonicos)
        self.alias = dict(ALIAS)
        if alias:
            self.alias.update(alias)
        # índice por clave normalizada -> nombre canónico
        self.por_norm = {}
        for c in self.canon:
            self.por_norm.setdefault(_norm(c), c)
        # alias también indexados por su forma normalizada
        self.alias_norm = {_norm(k): v for k, v in self.alias.items()}

    def resolve(self, name, fuzzy=False, cutoff=0.9):
        n = str(name).strip()
        if not n:
            return n
        # alias ANTES que el match exacto: corrige incluso grafías que el
        # propio dataset trae inconsistentes (p.ej. "Congo DR" -> "DR Congo").
        if n in self.alias:
            return self.alias[n]
        if n in self.canon:
            return n
        k = _norm(n)
        if k in self.alias_norm:
            return self.alias_norm[k]
        if k in self.por_norm:
            return self.por_norm[k]
        if fuzzy and len(k) >= 4:
            m = get_close_matches(k, list(self.por_norm), n=1, cutoff=cutoff)
            if m:
                return self.por_norm[m[0]]
        return n


@functools.lru_cache(maxsize=1)
def _default_resolver():
    """Resolver construido con los nombres reales del dataset (cacheado)."""
    canon = set()
    try:
        import pandas as pd
        d = pd.read_csv(CSV_DEFAULT, usecols=["home_team", "away_team"])
        canon = set(d["home_team"]) | set(d["away_team"])
    except Exception:
        pass
    return Resolver(canon)


def normaliza(name, fuzzy=False):
    """Normaliza un nombre al canónico del dataset (API simple para los módulos)."""
    return _default_resolver().resolve(name, fuzzy=fuzzy)
