"""Sincronización con el upstream (update_results.fusionar)."""
import pandas as pd

import update_results as up


def _df(rows):
    base = {"tournament": "Friendly", "city": None, "country": None, "neutral": False}
    return pd.DataFrame([{**base, **r} for r in rows])


ACTUAL = _df([
    # fixture futuro (sin marcador): siempre debe conservarse
    {"date": "2026-07-10", "home_team": "A", "away_team": "B",
     "home_score": None, "away_score": None},
    # resultado manual que el upstream NO tiene
    {"date": "2026-06-20", "home_team": "C", "away_team": "D",
     "home_score": 1.0, "away_score": 0.0},
    # partido que ambos tienen pero con marcador DISTINTO (corrupción)
    {"date": "2026-06-15", "home_team": "E", "away_team": "F",
     "home_score": 9.0, "away_score": 9.0},
])
UPSTREAM = _df([
    {"date": "2026-06-15", "home_team": "E", "away_team": "F",
     "home_score": 2.0, "away_score": 1.0},
    {"date": "2026-06-18", "home_team": "G", "away_team": "H",
     "home_score": 0.0, "away_score": 0.0},
])


def test_modo_normal_conserva_lo_local_y_upstream_corrige():
    out, stats, diffs, descartados = up.fusionar(ACTUAL, UPSTREAM, limpio=False)
    assert stats["fixtures_conservados"] == 1
    assert len(out) == 4  # fixture + manual + E-F corregido + G-H nuevo
    ef = out[(out["home_team"] == "E")].iloc[0]
    assert (ef["home_score"], ef["away_score"]) == (2.0, 1.0)  # upstream manda
    assert len(diffs) == 1          # el 9-9 vs 2-1 queda reportado
    assert len(descartados) == 1    # el manual C-D (informativo en modo normal)
    cd = out[(out["home_team"] == "C")]
    assert len(cd) == 1             # ...pero se conserva


def test_modo_limpio_descarta_resultados_no_upstream():
    out, stats, _, descartados = up.fusionar(ACTUAL, UPSTREAM, limpio=True)
    assert stats["descartados_con_marcador"] == 1
    assert out[(out["home_team"] == "C")].empty      # manual descartado
    assert not out[(out["home_team"] == "A")].empty  # fixture conservado
