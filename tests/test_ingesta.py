"""Ingesta de partidos del día y contraste (mundial.ingesta)."""
import numpy as np
import pandas as pd
import pytest

from mundial import ingesta as ing


def test_resultado_clase():
    assert ing.resultado_clase(2, 0) == 2   # gana local
    assert ing.resultado_clase(0, 1) == 0   # gana visitante
    assert ing.resultado_clase(1, 1) == 1   # empate
    assert np.isnan(ing.resultado_clase(None, 1))


def _scoreboard(completed=True):
    return {"events": [{
        "date": "2026-07-01T18:00Z",
        "competitions": [{
            "status": {"type": {"completed": completed}},
            "competitors": [
                {"homeAway": "home", "score": "2", "team": {"displayName": "Mexico"}},
                {"homeAway": "away", "score": "1", "team": {"displayName": "Congo DR"}},
            ]}]}]}


def test_parse_espn_scoreboard_completado():
    df = ing.parse_espn_scoreboard(_scoreboard(True))
    assert list(df.columns) == ing.COLS
    fila = df.iloc[0]
    assert fila["away_team"] == "DR Congo"      # alias de ESPN normalizado
    assert (fila["home_score"], fila["away_score"]) == (2.0, 1.0)


def test_parse_espn_scoreboard_en_juego_sin_marcador():
    df = ing.parse_espn_scoreboard(_scoreboard(False))
    assert pd.isna(df.iloc[0]["home_score"])


def test_a_filas_dataset_columnas_canonicas():
    df = ing.a_filas_dataset([{"date": "2026-07-01", "home_team": "A",
                               "away_team": "B", "home_score": 1, "away_score": 0}])
    assert list(df.columns) == ing.COLS
    assert ing.a_filas_dataset([]).empty


def test_merge_resultados_prioriza_lo_nuevo():
    base = pd.DataFrame([{"date": "2026-07-01", "home_team": "A", "away_team": "B",
                          "home_score": np.nan, "away_score": np.nan,
                          "tournament": "FIFA World Cup", "city": None,
                          "country": None, "neutral": True}])
    nuevo = base.copy()
    nuevo.loc[0, ["home_score", "away_score"]] = [3.0, 1.0]
    out = ing.merge_resultados(base, nuevo)
    assert len(out) == 1
    assert out.iloc[0]["home_score"] == 3.0


def test_evalua_contraste_metricas():
    log = pd.DataFrame([
        {"p_away": 0.2, "p_draw": 0.3, "p_home": 0.5, "res": 2},   # acierto
        {"p_away": 0.5, "p_draw": 0.3, "p_home": 0.2, "res": 2},   # fallo
        {"p_away": 0.3, "p_draw": 0.4, "p_home": 0.3, "res": np.nan},  # pendiente
    ])
    det, res = ing.evalua_contraste(log)
    assert res["n"] == 2                       # el pendiente no cuenta
    assert res["accuracy"] == pytest.approx(0.5)
    assert res["logloss"] > 0
