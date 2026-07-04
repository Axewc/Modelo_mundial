"""Métricas y value betting (mundial.backtest)."""
import numpy as np
import pytest

from mundial import backtest as bt


def test_rps_prediccion_perfecta_es_cero():
    P = np.array([[0, 0, 1.0], [1.0, 0, 0]])
    assert bt.rps_set(P, [2, 0]) == pytest.approx(0.0, abs=1e-9)


def test_rps_uniforme_valor_conocido():
    # uniforme, gana local: cumsum P=(1/3,2/3,1) vs Y=(0,0,1)
    # -> ((1/3)^2 + (2/3)^2) / 2 = 5/18
    P = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert bt.rps_set(P, [2]) == pytest.approx(5 / 18)


def test_logloss_y_brier_basicos():
    P = np.array([[0.1, 0.2, 0.7]])
    assert bt.logloss_set(P, [2]) == pytest.approx(-np.log(0.7))
    assert bt.brier_set(P, [2]) == pytest.approx(0.1**2 + 0.2**2 + 0.3**2)


def test_comparar_modelo_mercado_gana_el_mas_afilado():
    y = [2, 2, 0, 1]
    bueno = np.array([[0.1, 0.2, 0.7]] * 2 + [[0.7, 0.2, 0.1]] + [[0.2, 0.6, 0.2]])
    malo = np.full((4, 3), 1 / 3)
    tabla, frac = bt.comparar_modelo_mercado(bueno, malo, y)
    assert tabla.loc[tabla["Quién"] == "Modelo", "RPS"].iloc[0] < \
        tabla.loc[tabla["Quién"] == "Mercado", "RPS"].iloc[0]
    assert frac == 1.0  # el modelo gana partido a partido


def test_value_bets_flat_pnl():
    # única apuesta: local con edge 0.2, cuota 2.5; acierta -> +1.5 unidades
    model = np.array([[0.1, 0.2, 0.7]])
    market = np.array([[0.2, 0.3, 0.5]])
    cuotas = np.array([[5.0, 3.3, 2.5]])
    det, res = bt.value_bets(model, market, cuotas, [2], edge_min=0.05)
    assert res["n_apuestas"] == 1 and res["profit"] == pytest.approx(1.5)
    det2, res2 = bt.value_bets(model, market, cuotas, [0], edge_min=0.05)
    assert res2["profit"] == pytest.approx(-1.0)  # falla -> pierde el stake


def test_value_bets_respeta_edge_min():
    model = np.array([[0.34, 0.33, 0.33]])
    market = np.array([[0.33, 0.33, 0.34]])
    _, res = bt.value_bets(model, market, np.array([[3.0, 3.0, 3.0]]),
                           [0], edge_min=0.05)
    assert res["n_apuestas"] == 0


def test_kelly_cap():
    model = np.array([[0.05, 0.05, 0.90]])   # edge enorme
    market = np.array([[0.3, 0.3, 0.4]])
    det, res = bt.value_bets(model, market, np.array([[4.0, 4.0, 2.4]]),
                             [2], kelly_frac=1.0, kelly_cap=0.05)
    assert det.iloc[0]["Stake"] <= 0.05 + 1e-9


def test_clv():
    assert bt.clv(2.2, 2.0) == pytest.approx(0.1)
    assert bt.clv(1.9, 2.0) == pytest.approx(-0.05)
