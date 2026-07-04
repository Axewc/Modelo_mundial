"""Cuotas: implícita, margen, blend y parser de The Odds API (mundial.odds)."""
import numpy as np
import pytest

from mundial import odds


def test_implied_probs_orden_y_normalizacion():
    p = odds.implied_probs(1.50, 4.0, 7.0)  # [away, draw, home]
    assert p.sum() == pytest.approx(1.0)
    assert p[2] > p[1] > p[0]  # local favorito con cuota 1.50


def test_implied_probs_rechaza_cuotas_invalidas():
    with pytest.raises(ValueError):
        odds.implied_probs(1.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        odds.implied_probs(2.0, float("nan"), 4.0)


def test_overround_positivo_en_cuotas_reales():
    assert odds.overround(1.50, 4.0, 7.0) == pytest.approx(0.0595, abs=1e-3)


def test_blend_extremos_e_intermedio():
    model = [0.2, 0.3, 0.5]
    market = [0.4, 0.3, 0.3]
    assert np.allclose(odds.blend(model, market, 0.0), model)
    assert np.allclose(odds.blend(model, market, 1.0), market)
    mezcla = odds.blend(model, market, 0.5)
    assert mezcla.sum() == pytest.approx(1.0)
    assert market[0] > mezcla[0] > model[0] - 1e-12  # se mueve hacia el mercado


def _evento(home="Mexico", away="USA", books=None):
    books = books or [
        {"markets": [{"key": "h2h", "outcomes": [
            {"name": "Mexico", "price": 2.1}, {"name": "USA", "price": 3.4},
            {"name": "Draw", "price": 3.2}]}]},
        {"markets": [{"key": "h2h", "outcomes": [
            {"name": "Mexico", "price": 2.05}, {"name": "USA", "price": 3.5},
            {"name": "Draw", "price": 3.25}]}]},
    ]
    return {"commence_time": "2026-07-01T18:00:00Z", "home_team": home,
            "away_team": away, "bookmakers": books}


def test_parse_eventos_promedia_casas_y_normaliza_nombres():
    df = odds.parse_eventos([_evento()])
    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["away_team"] == "United States"  # USA normalizado
    assert fila["n_casas"] == 2
    assert fila["cuota_home"] == pytest.approx(np.median([2.1, 2.05]), abs=0.01)
    assert fila["p_home"] + fila["p_draw"] + fila["p_away"] == pytest.approx(1.0)


def test_parse_eventos_descarta_sin_mercado_h2h():
    ev = _evento(books=[{"markets": [{"key": "totals", "outcomes": []}]}])
    assert odds.parse_eventos([ev]).empty


def test_buscar_partido_en_ambas_orientaciones():
    df = odds.parse_eventos([_evento()])
    assert odds.buscar_partido(df, "Mexico", "United States") is not None
    assert odds.buscar_partido(df, "United States", "Mexico") is not None
    assert odds.buscar_partido(df, "Brazil", "Spain") is None
    assert odds.buscar_partido(None, "A", "B") is None
