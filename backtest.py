"""
backtest.py — ¿Le gana el modelo al mercado?
============================================
La única validación que de verdad importa para un modelo con ángulo de
trading: contrastar el modelo contra las CUOTAS DE CIERRE del mercado.

Hace tres cosas:
  1. Puntúa modelo vs mercado con la MISMA métrica (RPS/LogLoss/Brier) sobre
     los mismos partidos. Si el mercado gana, no hay edge propio aún.
  2. Simula apostar SOLO cuando el modelo ve "valor" (más prob que el
     mercado) — staking plano y Kelly fraccional — y reporta ROI.
     Como se apuesta A LAS CUOTAS DE CIERRE, un ROI>0 ya es señal fuerte
     (batir la línea de cierre es difícil).
  3. Predicciones POINT-IN-TIME (walk-forward): para cada partido, el modelo
     se entrena solo con datos ANTERIORES. Nada de mirar el futuro.

Convención de orden en todos los vectores: [away, draw, home] = CLASSES.
"""
import numpy as np
import pandas as pd

try:
    from ingesta import normaliza_equipo as _norm
except Exception:
    def _norm(n):
        return str(n).strip()


# ============================================================
# Métricas (numpy puro, testeable sin sklearn)
# ============================================================
def _onehot(outcomes):
    Y = np.zeros((len(outcomes), 3))
    for i, o in enumerate(outcomes):
        Y[i, int(o)] = 1.0
    return Y


def rps_set(P, outcomes):
    """Ranked Probability Score medio (menor = mejor). P en orden [away,draw,home]."""
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    Y = _onehot(outcomes)
    cp, cy = np.cumsum(P, 1), np.cumsum(Y, 1)
    return float(np.mean(np.sum((cp - cy) ** 2, 1) / 2))


def logloss_set(P, outcomes):
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    idx = np.arange(len(outcomes))
    return float(-np.mean(np.log(P[idx, np.asarray(outcomes, int)])))


def brier_set(P, outcomes):
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    Y = _onehot(outcomes)
    return float(np.mean(np.sum((P - Y) ** 2, 1)))


def comparar_modelo_mercado(model_P, market_P, outcomes):
    """Tabla modelo vs mercado en las MISMAS métricas + cuántas veces el
    modelo gana al mercado por partido (RPS individual)."""
    rps_m = _rps_individual(model_P, outcomes)
    rps_k = _rps_individual(market_P, outcomes)
    return pd.DataFrame([
        {"Quién": "Modelo", "RPS": rps_set(model_P, outcomes),
         "LogLoss": logloss_set(model_P, outcomes), "Brier": brier_set(model_P, outcomes)},
        {"Quién": "Mercado", "RPS": rps_set(market_P, outcomes),
         "LogLoss": logloss_set(market_P, outcomes), "Brier": brier_set(market_P, outcomes)},
    ]).round(4), float(np.mean(rps_m < rps_k))


def _rps_individual(P, outcomes):
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    Y = _onehot(outcomes)
    cp, cy = np.cumsum(P, 1), np.cumsum(Y, 1)
    return np.sum((cp - cy) ** 2, 1) / 2


# ============================================================
# Closing Line Value (cuando hay dos tomas de cuota)
# ============================================================
def clv(odds_bet, odds_close):
    """CLV = odds_apostada / odds_cierre - 1. Positivo = conseguiste mejor
    cuota que la de cierre (señal de edge a largo plazo). Requiere AMBAS
    tomas; en un backtest a cierre, CLV≈0 por construcción."""
    return float(np.asarray(odds_bet, float) / np.asarray(odds_close, float) - 1)


# ============================================================
# Simulación de apuestas de valor
# ============================================================
def value_bets(model_P, market_P, odds, outcomes, edge_min=0.03,
               kelly_frac=0.0, kelly_cap=0.05):
    """Apuesta el desenlace con MAYOR ventaja del modelo sobre el mercado,
    si esa ventaja >= edge_min. odds = cuotas decimales [away,draw,home].

    kelly_frac=0 -> staking plano (1 unidad por apuesta).
    kelly_frac>0 -> Kelly fraccional (stake = kelly_frac * f*, capado a kelly_cap).
    Devuelve (detalle_df, resumen_dict).
    """
    model_P = np.asarray(model_P, float)
    market_P = np.asarray(market_P, float)
    odds = np.asarray(odds, float)
    outcomes = np.asarray(outcomes, int)
    nombres = np.array(["Visitante", "Empate", "Local"])

    filas = []
    staked = profit = 0.0
    for i in range(len(outcomes)):
        edges = model_P[i] - market_P[i]
        j = int(np.argmax(edges))           # desenlace con mayor valor
        if edges[j] < edge_min:
            continue
        o = odds[i, j]
        if not np.isfinite(o) or o <= 1:
            continue
        b = o - 1.0
        p = float(model_P[i, j])
        if kelly_frac > 0:
            f = (b * p - (1 - p)) / b        # fracción de Kelly
            stake = float(np.clip(kelly_frac * f, 0.0, kelly_cap))
            if stake <= 0:
                continue
        else:
            stake = 1.0
        gano = (outcomes[i] == j)
        pl = stake * b if gano else -stake
        staked += stake
        profit += pl
        filas.append({"i": i, "Apuesta": nombres[j], "Cuota": round(o, 2),
                      "P_modelo": round(p, 3), "P_mercado": round(float(market_P[i, j]), 3),
                      "Edge": round(float(edges[j]), 3), "Stake": round(stake, 3),
                      "Acertó": bool(gano), "P&L": round(pl, 3)})
    det = pd.DataFrame(filas)
    n = len(det)
    resumen = {
        "n_apuestas": n,
        "aciertos": int(det["Acertó"].sum()) if n else 0,
        "hit_rate": float(det["Acertó"].mean()) if n else 0.0,
        "staked": round(staked, 2),
        "profit": round(profit, 2),
        "ROI": round(profit / staked, 4) if staked > 0 else 0.0,
    }
    return det, resumen


# ============================================================
# Walk-forward: predicciones POINT-IN-TIME del modelo (Dixon-Coles)
# ============================================================
def _market_en_orientacion(o, home, away):
    """Devuelve (market_P[away,draw,home], odds[away,draw,home]) en la
    orientacion del partido del dataset (home/away), reordenando si la fila
    de cuotas viene al reves."""
    from odds import implied_probs
    oh = float(o.get("cuota_home", o.get("home_odds", np.nan)))
    od = float(o.get("cuota_draw", o.get("draw_odds", np.nan)))
    oa = float(o.get("cuota_away", o.get("away_odds", np.nan)))
    if _norm(o.get("home_team", "")) == _norm(home):
        mh, md, ma = oh, od, oa
    else:                       # la fila de cuotas esta invertida
        mh, md, ma = oa, od, oh
    P = implied_probs(mh, md, ma)          # [away, draw, home]
    return P, np.array([ma, md, mh], float)


def construir_backtest(art, odds_df, retrain_every=15, min_train=200,
                       max_dias=3):
    """Une partidos JUGADOS del dataset con sus cuotas y genera, en
    walk-forward, las probabilidades del modelo (Dixon-Coles point-in-time).

    odds_df: columnas date, home_team, away_team y cuota_home/draw/away
    (o home_odds/draw_odds/away_odds). Devuelve dict con arrays alineados:
    model_P, market_P, odds, outcomes y un DataFrame de detalle.
    """
    import modelo_core as mc

    dfp = art["df"].reset_index(drop=True).copy()
    dfp["_H"] = dfp["home_team"].map(_norm)
    dfp["_A"] = dfp["away_team"].map(_norm)

    od = odds_df.copy()
    od["_H"] = od["home_team"].map(_norm)
    od["_A"] = od["away_team"].map(_norm)
    od["_d"] = pd.to_datetime(od["date"]) if "date" in od.columns else pd.NaT

    seleccion = []   # (pos, home, away, neutral, outcome, market_P, odds_row)
    for pos, row in dfp.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        cand = od[((od["_H"] == row["_H"]) & (od["_A"] == row["_A"])) |
                  ((od["_H"] == row["_A"]) & (od["_A"] == row["_H"]))]
        if cand.empty:
            continue
        if cand["_d"].notna().any() and pd.notna(row["date"]):
            cand = cand.assign(_dd=(cand["_d"] - row["date"]).abs())
            cand = cand[cand["_dd"] <= pd.Timedelta(days=max_dias)]
            if cand.empty:
                continue
            o = cand.sort_values("_dd").iloc[0]
        else:
            o = cand.iloc[0]
        try:
            mP, oo = _market_en_orientacion(o, row["home_team"], row["away_team"])
        except Exception:
            continue
        if row["home_score"] > row["away_score"]:
            outcome = 2
        elif row["home_score"] < row["away_score"]:
            outcome = 0
        else:
            outcome = 1
        seleccion.append((pos, row["home_team"], row["away_team"],
                          bool(row["neutral_match"]), outcome, mP, oo))

    seleccion.sort(key=lambda x: x[0])

    model_P, market_P, odds_arr, outcomes, det = [], [], [], [], []
    model = None
    last_train = -10 ** 9
    for (pos, home, away, neutral, outcome, mP, oo) in seleccion:
        if pos < min_train:
            continue
        if model is None or (pos - last_train) >= retrain_every:
            train = dfp.iloc[:pos]
            if len(train) >= min_train:
                try:
                    model = mc.fit_dc(train, alpha=0.05)
                    last_train = pos
                except Exception:
                    model = None
        if model is None:
            continue
        try:
            p = mc.dc_result_probs(model, home, away, neutral)
        except Exception:
            continue
        model_P.append(p); market_P.append(mP); odds_arr.append(oo)
        outcomes.append(outcome)
        det.append({"date": dfp.loc[pos, "date"], "Local": home, "Visitante": away,
                    "Resultado": ["V", "E", "L"][outcome]})

    return {
        "model_P": np.array(model_P), "market_P": np.array(market_P),
        "odds": np.array(odds_arr), "outcomes": np.array(outcomes, int),
        "detalle": pd.DataFrame(det), "n": len(outcomes),
    }
