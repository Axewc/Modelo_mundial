"""Funciones puras del núcleo (mundial.modelo_core) y de forma (analisis_forma)."""
import numpy as np
import pandas as pd
import pytest

from mundial import modelo_core as mc
from mundial import analisis_forma as af


# ---------- prob_avance (eliminatorias con prórroga y penales) ----------
def test_prob_avance_suma_uno_y_favorece_al_mejor():
    p1x2 = [0.30, 0.30, 0.40]          # [away, draw, home]
    av = mc.prob_avance(p1x2, lh=1.2, la=1.1)
    assert av.sum() == pytest.approx(1.0)
    assert av[1] > 0.5                 # el favorito avanza más de la mitad


def test_prob_avance_shootout_extremo():
    # si el local ganara SIEMPRE los penales, su prob de avance sube
    p = [0.30, 0.40, 0.30]
    con = mc.prob_avance(p, 1.0, 1.0, shootout=1.0, prorroga=False)
    sin = mc.prob_avance(p, 1.0, 1.0, shootout=0.0, prorroga=False)
    assert con[1] == pytest.approx(0.70)
    assert sin[1] == pytest.approx(0.30)


# ---------- binomial negativa ≈ Poisson cuando r es grande ----------
def test_nb_converge_a_poisson():
    pred = {"lambdas": lambda h, a, n: (1.4, 1.1), "rho": -0.05}
    M_dc = mc.dc_score_matrix(pred, "X", "Y", True)
    M_nb = mc.nb_score_matrix(pred, "X", "Y", True, r=10_000)
    assert np.abs(M_dc - M_nb).max() < 1e-3


# ---------- mercados agrupados sobre una matriz conocida ----------
def test_mercados_agrupados_matriz_2x2():
    # matriz 2x2: 0-0=0.4, 0-1=0.1, 1-0=0.3, 1-1=0.2
    M = np.array([[0.4, 0.1], [0.3, 0.2]])
    mk = af.mercados_agrupados(M)
    assert mk["btts"] == pytest.approx(0.2)
    assert mk["local_2plus"] == pytest.approx(0.0)
    assert mk["doble_1x"] == pytest.approx(0.3 + 0.6)  # local + empates
    top = af.marcador_top_por_resultado(M)
    assert top["empate"]["marcador"] == "0–0"
    assert top["local"]["marcador"] == "1–0"


# ---------- peso por competencia en el análisis de forma ----------
def test_peso_competencia():
    assert af._peso_competencia("Friendly") == pytest.approx(0.55)
    assert af._peso_competencia("FIFA World Cup qualification") == pytest.approx(0.90)
    assert af._peso_competencia("FIFA World Cup") == pytest.approx(1.0)


# ---------- load_data: fusión de partidos extra y dedup quirúrgico ----------
def _csv_temporal(tmp_path):
    df = pd.DataFrame([
        {"date": "2026-06-01", "home_team": "Spain", "away_team": "France",
         "home_score": 1.0, "away_score": 0.0, "tournament": "Friendly",
         "city": None, "country": None, "neutral": True},
        {"date": "2026-07-10", "home_team": "Spain", "away_team": "Brazil",
         "home_score": None, "away_score": None, "tournament": "FIFA World Cup",
         "city": None, "country": None, "neutral": True},
    ])
    p = tmp_path / "mini.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_load_data_extra_convierte_fixture_en_jugado(tmp_path):
    path = _csv_temporal(tmp_path)
    extra = pd.DataFrame([{"date": "2026-07-10", "home_team": "Spain",
                           "away_team": "Brazil", "home_score": 2.0,
                           "away_score": 2.0, "tournament": "FIFA World Cup",
                           "neutral": True}])
    jugados0, pend0 = mc.load_data(path)
    jugados1, pend1 = mc.load_data(path, extra=extra)
    assert len(jugados1) == len(jugados0) + 1
    assert len(pend1) == len(pend0) - 1
