"""
modelo_core.py
==============
Nucleo del modelo predictivo del Mundial 2026, extraido del notebook
MODELO_1_2.ipynb para poder reutilizarlo desde una app de Streamlit.

Contiene todo el pipeline:
  - Ingesta limpia (jugados vs por jugar)
  - ELO V3 (historia completa)
  - Forma reciente + H2H
  - Poisson (42 features), XGBoost (calibrado) y Dixon-Coles
  - Evaluacion comparativa (LogLoss / Brier / Accuracy)
  - Simulador del torneo completo

XGBoost es OPCIONAL: si no esta instalado, la app sigue funcionando con
los demas metodos.
"""

import collections
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _calibrar(modelo, X_val, y_val):
    """Calibra un modelo ya entrenado. Usa FrozenEstimator (sklearn >=1.6)
    y cae a cv='prefit' en versiones antiguas."""
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic")
    except Exception:
        cal = CalibratedClassifierCV(modelo, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    return cal

CSV_DEFAULT = "results_actualizado_mundial_2026.csv"
ANIO_INICIO = 2018
RANDOM_STATE = 42
CLASSES = [0, 1, 2]  # 0=gana visitante, 1=empate, 2=gana local

# Ventana de recencia (dias) compartida por ELO y Dixon-Coles.
# Mas grande -> pesa mas la historia/reputacion (Brasil, Alemania suben);
# mas pequeña -> pesa mas la forma actual. 2920 = 8 años (balance realista).
TAU_DAYS = 2920

PESO_TORNEO = {
    "FIFA World Cup": 2.0, "UEFA Euro": 1.8, "Copa América": 1.7,
    "FIFA World Cup qualification": 1.5, "UEFA Nations League": 1.3,
    "CONCACAF Nations League": 1.2, "African Cup of Nations": 1.2,
    "AFC Asian Cup": 1.2, "Gold Cup": 1.3, "Friendly": 0.6,
}
COMP_FUERTE = ["FIFA World Cup", "UEFA Euro", "Copa América",
               "African Cup of Nations", "AFC Asian Cup", "Gold Cup"]


# ============================================================
# 1) INGESTA
# ============================================================
def load_data(path=CSV_DEFAULT):
    df_raw = pd.read_csv(path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    tiene = df_raw["home_score"].notna() & df_raw["away_score"].notna()
    completos = df_raw[["date", "home_team", "away_team"]].notna().all(axis=1)
    df_full = (df_raw[tiene & completos]
               .sort_values("date").reset_index(drop=True).copy())
    por_jugar = (df_raw[~tiene & completos]
                 .sort_values("date").reset_index(drop=True).copy())
    return df_full, por_jugar


# ============================================================
# 2) ELO V3 (historia completa)
# ============================================================
def compute_elo(df_full):
    d = df_full.copy()
    d["neutral"] = d["neutral"].astype(str).str.lower()
    d["neutral_match"] = d["neutral"].isin(["true", "1"]).astype(int)
    d["home_advantage"] = 1 - d["neutral_match"]
    d["peso_torneo"] = d["tournament"].map(PESO_TORNEO).fillna(1.0)

    fecha_maxima = d["date"].max()
    equipos = pd.unique(pd.concat([d["home_team"], d["away_team"]]))
    elo = {e: 1500.0 for e in equipos}
    elo_home, elo_away = [], []

    for _, row in d.iterrows():
        h, a = row["home_team"], row["away_team"]
        eh, ea = elo[h], elo[a]
        elo_home.append(eh); elo_away.append(ea)
        exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))

        if row["home_score"] > row["away_score"]:
            sh, sa = 1, 0
            dr = eh - ea          # ventaja ELO del ganador (local)
        elif row["home_score"] < row["away_score"]:
            sh, sa = 0, 1
            dr = ea - eh          # ventaja ELO del ganador (visitante)
        else:
            sh, sa = 0.5, 0.5
            dr = 0.0

        gd = abs(row["home_score"] - row["away_score"])
        # Margen con autocorreccion (estilo FiveThirtyEight): golear a un
        # rival muy inferior NO infla el rating -> frena el sesgo de
        # selecciones que acumulan goleadas regionales contra rivales debiles.
        mov = np.log(gd + 1) * (2.2 / (0.001 * dr + 2.2))
        pr = np.exp(-(fecha_maxima - row["date"]).days / TAU_DAYS)

        t = row["tournament"]
        if t == "Friendly":
            pc = 0.45
        elif "qualification" in str(t).lower():
            pc = 0.85
        elif t in COMP_FUERTE:
            pc = 1.15
        else:
            pc = 0.75

        k = 24 * pr * pc * mov
        elo[h] += np.clip(k * (sh - exp_home), -32, 32)
        elo[a] += np.clip(k * (sa - (1 - exp_home)), -32, 32)

    d["elo_home"] = elo_home
    d["elo_away"] = elo_away
    d["elo_diff"] = d["elo_home"] - d["elo_away"]

    df = d[d["date"] >= f"{ANIO_INICIO}-01-01"].reset_index(drop=True).copy()
    return df, elo


# ============================================================
# 3) FORMA + H2H + dataset de entrenamiento
# ============================================================
def _resumen(partidos):
    if not partidos:
        return {"gf": 0, "gc": 0, "wins": 0, "draws": 0, "losses": 0,
                "gd": 0, "streak": 0}
    gf = sum(x["gf"] for x in partidos)
    gc = sum(x["gc"] for x in partidos)
    wins = sum(x["resultado"] == "W" for x in partidos)
    draws = sum(x["resultado"] == "D" for x in partidos)
    losses = sum(x["resultado"] == "L" for x in partidos)
    streak = 0
    for p in reversed(partidos):
        if p["resultado"] == "W":
            streak += 1
        else:
            break
    return {"gf": gf, "gc": gc, "wins": wins, "draws": draws,
            "losses": losses, "gd": gf - gc, "streak": streak}


def _obtener_h2h(hist, home, away):
    hh = ha = hd = 0
    for x in hist[-5:]:
        if x["winner"] == home:
            hh += 1
        elif x["winner"] == away:
            ha += 1
        else:
            hd += 1
    return hh, ha, hd


def build_dataset(df):
    historial, h2h, registros = {}, {}, []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        historial.setdefault(home, [])
        historial.setdefault(away, [])
        h5 = _resumen(historial[home][-5:]); h10 = _resumen(historial[home][-10:])
        a5 = _resumen(historial[away][-5:]); a10 = _resumen(historial[away][-10:])
        clave = tuple(sorted([home, away]))
        hh, ha, hd = _obtener_h2h(h2h.get(clave, []), home, away)

        if row["home_score"] > row["away_score"]:
            resultado = 2
        elif row["home_score"] < row["away_score"]:
            resultado = 0
        else:
            resultado = 1

        registros.append({
            "elo_home": row["elo_home"], "elo_away": row["elo_away"],
            "elo_diff": row["elo_diff"], "home_advantage": row["home_advantage"],
            "neutral_match": row["neutral_match"], "peso_torneo": row["peso_torneo"],
            "gf5_home": h5["gf"], "gc5_home": h5["gc"], "wins5_home": h5["wins"],
            "draws5_home": h5["draws"], "losses5_home": h5["losses"], "gd5_home": h5["gd"],
            "gf10_home": h10["gf"], "gc10_home": h10["gc"], "wins10_home": h10["wins"],
            "draws10_home": h10["draws"], "losses10_home": h10["losses"], "gd10_home": h10["gd"],
            "gf5_away": a5["gf"], "gc5_away": a5["gc"], "wins5_away": a5["wins"],
            "draws5_away": a5["draws"], "losses5_away": a5["losses"], "gd5_away": a5["gd"],
            "gf10_away": a10["gf"], "gc10_away": a10["gc"], "wins10_away": a10["wins"],
            "draws10_away": a10["draws"], "losses10_away": a10["losses"], "gd10_away": a10["gd"],
            "streak_home": h5["streak"], "streak_away": a5["streak"],
            "h2h_home": hh, "h2h_away": ha, "h2h_draw": hd,
            "home_score": row["home_score"], "away_score": row["away_score"],
            "resultado": resultado,
        })

        if row["home_score"] > row["away_score"]:
            rh, ra, winner = "W", "L", home
        elif row["home_score"] < row["away_score"]:
            rh, ra, winner = "L", "W", away
        else:
            rh, ra, winner = "D", "D", "DRAW"
        historial[home].append({"gf": row["home_score"], "gc": row["away_score"], "resultado": rh})
        historial[away].append({"gf": row["away_score"], "gc": row["home_score"], "resultado": ra})
        historial[home] = historial[home][-10:]
        historial[away] = historial[away][-10:]
        h2h.setdefault(clave, []).append({"winner": winner})

    dm = pd.DataFrame(registros).fillna(0)
    for c in ["gf5", "gf10", "gd5", "gd10", "wins5", "wins10"]:
        dm[f"{c}_diff"] = dm[f"{c}_home"] - dm[f"{c}_away"]
    dm["streak_diff"] = dm["streak_home"] - dm["streak_away"]
    return dm, historial, h2h


FEATURES = [
    "elo_home", "elo_away", "elo_diff", "home_advantage", "neutral_match", "peso_torneo",
    "gf5_home", "gc5_home", "wins5_home", "draws5_home", "losses5_home", "gd5_home",
    "gf10_home", "gc10_home", "wins10_home", "draws10_home", "losses10_home", "gd10_home",
    "gf5_away", "gc5_away", "wins5_away", "draws5_away", "losses5_away", "gd5_away",
    "gf10_away", "gc10_away", "wins10_away", "draws10_away", "losses10_away", "gd10_away",
    "streak_home", "streak_away", "h2h_home", "h2h_away", "h2h_draw",
    "gf5_diff", "gf10_diff", "gd5_diff", "gd10_diff", "wins5_diff", "wins10_diff", "streak_diff",
]


# ============================================================
# 4) DIXON-COLES
# ============================================================
def _pesos_dc(d):
    fm = d["date"].max()
    pr = np.exp(-(fm - d["date"]).dt.days / TAU_DAYS)
    t = d["tournament"].astype(str)
    pc = np.where(t.eq("Friendly"), 0.45,
         np.where(t.str.lower().str.contains("qualification"), 0.85,
         np.where(t.isin(COMP_FUERTE), 1.15, 0.75)))
    return pr.values * pc


def fit_dc(d, alpha=0.05):
    d = d.reset_index(drop=True)
    teams = sorted(set(d["home_team"]) | set(d["away_team"]))
    team_col = pd.concat([d["home_team"], d["away_team"]], ignore_index=True)
    opp_col = pd.concat([d["away_team"], d["home_team"]], ignore_index=True)
    goals = pd.concat([d["home_score"], d["away_score"]], ignore_index=True).astype(int)
    is_home = np.concatenate([(1 - d["neutral_match"].values),
                              np.zeros(len(d), dtype=int)])
    atk = pd.get_dummies(team_col).reindex(columns=teams, fill_value=0).add_prefix("atk_")
    dfn = pd.get_dummies(opp_col).reindex(columns=teams, fill_value=0).add_prefix("dfn_")
    X = pd.concat([atk, dfn, pd.Series(is_home, name="is_home")], axis=1).astype(float)
    w = _pesos_dc(d)
    m = PoissonRegressor(alpha=alpha, max_iter=2000)
    m.fit(X, goals.values, sample_weight=np.concatenate([w, w]))
    coef = dict(zip(X.columns, m.coef_))
    c = float(m.intercept_)
    atk_d = {t: coef.get("atk_" + t, 0.0) for t in teams}
    dfn_d = {t: coef.get("dfn_" + t, 0.0) for t in teams}
    gamma = coef.get("is_home", 0.0)

    def lambdas(home, away, neutral):
        hb = 0 if neutral else 1
        lh = np.exp(c + atk_d.get(home, 0.0) + dfn_d.get(away, 0.0) + gamma * hb)
        la = np.exp(c + atk_d.get(away, 0.0) + dfn_d.get(home, 0.0))
        return lh, la

    atk_h = d["home_team"].map(atk_d).values; dfn_h = d["home_team"].map(dfn_d).values
    atk_a = d["away_team"].map(atk_d).values; dfn_a = d["away_team"].map(dfn_d).values
    hb = (1 - d["neutral_match"].values)
    lh = np.exp(c + atk_h + dfn_a + gamma * hb)
    la = np.exp(c + atk_a + dfn_h)
    xs = d["home_score"].values.astype(int); ys = d["away_score"].values.astype(int)
    m00 = (xs == 0) & (ys == 0); m01 = (xs == 0) & (ys == 1)
    m10 = (xs == 1) & (ys == 0); m11 = (xs == 1) & (ys == 1)

    def negll(rho):
        tau = np.ones(len(d))
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))

    rho = float(minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded").x)
    return {"c": c, "atk": atk_d, "dfn": dfn_d, "gamma": gamma,
            "rho": rho, "lambdas": lambdas, "teams": teams}


def dc_score_matrix(pred, home, away, neutral, max_goals=10):
    lh, la = pred["lambdas"](home, away, neutral)
    rho = pred["rho"]
    h = poisson.pmf(np.arange(max_goals + 1), lh)
    a = poisson.pmf(np.arange(max_goals + 1), la)
    M = np.outer(h, a)
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


def dc_result_probs(pred, home, away, neutral, max_goals=10):
    M = dc_score_matrix(pred, home, away, neutral, max_goals)
    return np.array([np.triu(M, 1).sum(), np.trace(M), np.tril(M, -1).sum()])


def poisson_result_probs(lh, la, max_goals=10):
    h = poisson.pmf(np.arange(max_goals + 1), lh)
    a = poisson.pmf(np.arange(max_goals + 1), la)
    M = np.outer(h, a)
    p_home = np.tril(M, -1).sum(); p_draw = np.trace(M); p_away = np.triu(M, 1).sum()
    s = p_home + p_draw + p_away
    return np.array([p_away / s, p_draw / s, p_home / s])


# ============================================================
# 5) PIPELINE COMPLETO
# ============================================================
def _brier(y, proba):
    y = np.asarray(y)
    Y = np.zeros((len(y), 3))
    for i, c in enumerate(CLASSES):
        Y[:, i] = (y == c)
    return np.mean(np.sum((proba - Y) ** 2, axis=1))


def _evaluar(nombre, y, proba):
    proba = np.clip(np.asarray(proba, float), 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[np.argmax(proba, axis=1)]
    return {"Modelo": nombre, "LogLoss": log_loss(y, proba, labels=CLASSES),
            "Brier": _brier(y, proba), "Accuracy": accuracy_score(y, pred)}


def run_pipeline(path=CSV_DEFAULT):
    df_full, por_jugar = load_data(path)
    df, elo = compute_elo(df_full)
    dm, historial, h2h = build_dataset(df)

    split = int(len(dm) * 0.80)
    train, test = dm.iloc[:split], dm.iloc[split:]
    Xtr, Xte = train[FEATURES], test[FEATURES]
    yr_tr, yr_te = train["resultado"], test["resultado"]
    y_true = yr_te.values

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    m_home = PoissonRegressor(alpha=0.1, max_iter=500).fit(Xtr_s, train["home_score"])
    m_away = PoissonRegressor(alpha=0.1, max_iter=500).fit(Xtr_s, train["away_score"])
    pred_home, pred_away = m_home.predict(Xte_s), m_away.predict(Xte_s)

    evals = []
    # baseline mayoria
    p_clase = np.array([np.mean(yr_tr == c) for c in CLASSES])
    evals.append(_evaluar("Baseline mayoria", y_true, np.tile(p_clase, (len(y_true), 1))))
    # baseline ELO
    fb = ["elo_diff", "home_advantage", "neutral_match"]
    blr = LogisticRegression(max_iter=1000).fit(train[fb], yr_tr)
    pb = blr.predict_proba(test[fb])[:, np.argsort(blr.classes_)]
    evals.append(_evaluar("Baseline ELO", y_true, pb))
    # poisson MC
    pp = np.array([poisson_result_probs(lh, la) for lh, la in zip(pred_home, pred_away)])
    evals.append(_evaluar("Poisson (MC)", y_true, pp))

    m_xgb = m_xgb_cal = None
    if HAS_XGB:
        vcut = int(len(Xtr) * 0.85)
        m_xgb = XGBClassifier(
            n_estimators=2000, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, gamma=0.2,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            early_stopping_rounds=50, random_state=RANDOM_STATE)
        m_xgb.fit(Xtr.iloc[:vcut], yr_tr.iloc[:vcut],
                  eval_set=[(Xtr.iloc[vcut:], yr_tr.iloc[vcut:])], verbose=False)
        m_xgb_cal = _calibrar(m_xgb, Xtr.iloc[vcut:], yr_tr.iloc[vcut:])
        evals.append(_evaluar("XGBoost", y_true, m_xgb.predict_proba(Xte)))
        evals.append(_evaluar("XGBoost calibrado", y_true, m_xgb_cal.predict_proba(Xte)))

    # dixon-coles eval + full
    dc_eval = fit_dc(df.iloc[:split])
    pdc = np.array([dc_result_probs(dc_eval, h, a, bool(n)) for h, a, n in
                    zip(df.iloc[split:]["home_team"], df.iloc[split:]["away_team"],
                        df.iloc[split:]["neutral_match"])])
    evals.append(_evaluar("Dixon-Coles", y_true, pdc))
    dc_full = fit_dc(df)

    tabla_eval = (pd.DataFrame(evals).sort_values("LogLoss")
                  .reset_index(drop=True).round(4))

    # snapshot de forma actual (para el predictor)
    eq = []
    for t in historial:
        u5 = _resumen(historial[t][-5:]); u10 = _resumen(historial[t][-10:])
        eq.append({"PAIS": t, "ELO": elo.get(t, 1500),
                   "GF5": u5["gf"], "GC5": u5["gc"], "W5": u5["wins"], "D5": u5["draws"],
                   "L5": u5["losses"], "GF10": u10["gf"], "GC10": u10["gc"], "W10": u10["wins"],
                   "D10": u10["draws"], "L10": u10["losses"], "STREAK": u5["streak"]})
    equipos_actuales = pd.DataFrame(eq).sort_values("ELO", ascending=False).reset_index(drop=True)

    ranking = (pd.DataFrame({"PAIS": list(elo), "ELO": list(elo.values())})
               .sort_values("ELO", ascending=False).reset_index(drop=True).round(1))

    return {
        "df": df, "por_jugar": por_jugar, "elo": elo, "ranking": ranking,
        "scaler": scaler, "m_home": m_home, "m_away": m_away,
        "m_xgb": m_xgb, "m_xgb_cal": m_xgb_cal, "dc_full": dc_full,
        "h2h": h2h, "equipos_actuales": equipos_actuales,
        "tabla_eval": tabla_eval, "has_xgb": HAS_XGB,
        "mae_home": float(np.mean(np.abs(pred_home - test["home_score"].values))),
        "mae_away": float(np.mean(np.abs(pred_away - test["away_score"].values))),
    }


# ============================================================
# 6) PREDECIR UN PARTIDO (todos los metodos)
# ============================================================
def _features_partido(art, local, visitante, neutral_match, peso_torneo):
    eq = art["equipos_actuales"]
    dl = eq[eq["PAIS"] == local].iloc[0]
    dv = eq[eq["PAIS"] == visitante].iloc[0]
    row = {
        "elo_home": dl["ELO"], "elo_away": dv["ELO"], "elo_diff": dl["ELO"] - dv["ELO"],
        "home_advantage": 1 - neutral_match, "neutral_match": neutral_match,
        "peso_torneo": peso_torneo,
        "gf5_home": dl["GF5"], "gc5_home": dl["GC5"], "wins5_home": dl["W5"],
        "draws5_home": dl["D5"], "losses5_home": dl["L5"], "gd5_home": dl["GF5"] - dl["GC5"],
        "gf10_home": dl["GF10"], "gc10_home": dl["GC10"], "wins10_home": dl["W10"],
        "draws10_home": dl["D10"], "losses10_home": dl["L10"], "gd10_home": dl["GF10"] - dl["GC10"],
        "gf5_away": dv["GF5"], "gc5_away": dv["GC5"], "wins5_away": dv["W5"],
        "draws5_away": dv["D5"], "losses5_away": dv["L5"], "gd5_away": dv["GF5"] - dv["GC5"],
        "gf10_away": dv["GF10"], "gc10_away": dv["GC10"], "wins10_away": dv["W10"],
        "draws10_away": dv["D10"], "losses10_away": dv["L10"], "gd10_away": dv["GF10"] - dv["GC10"],
        "streak_home": dl["STREAK"], "streak_away": dv["STREAK"],
    }
    clave = tuple(sorted([local, visitante]))
    hh, ha, hd = _obtener_h2h(art["h2h"].get(clave, []), local, visitante)
    row.update({"h2h_home": hh, "h2h_away": ha, "h2h_draw": hd,
                "gf5_diff": row["gf5_home"] - row["gf5_away"],
                "gf10_diff": row["gf10_home"] - row["gf10_away"],
                "gd5_diff": row["gd5_home"] - row["gd5_away"],
                "gd10_diff": row["gd10_home"] - row["gd10_away"],
                "wins5_diff": row["wins5_home"] - row["wins5_away"],
                "wins10_diff": row["wins10_home"] - row["wins10_away"],
                "streak_diff": row["streak_home"] - row["streak_away"]})
    return pd.DataFrame([row])[FEATURES]


def predict_match(art, local, visitante, neutral_match, peso_torneo):
    X = _features_partido(art, local, visitante, neutral_match, peso_torneo)
    Xs = art["scaler"].transform(X)
    lam_l = float(art["m_home"].predict(Xs)[0])
    lam_v = float(art["m_away"].predict(Xs)[0])
    res = {"metodos": {}, "goles": {}}
    res["goles"]["Poisson(42)"] = (round(lam_l, 3), round(lam_v, 3))

    res["metodos"]["Poisson + MC"] = poisson_result_probs(lam_l, lam_v)

    if art["has_xgb"]:
        res["metodos"]["XGBoost (sin calibrar)"] = art["m_xgb"].predict_proba(X)[0]
        res["metodos"]["XGBoost (calibrado)"] = art["m_xgb_cal"].predict_proba(X)[0]

    res["metodos"]["Dixon-Coles"] = dc_result_probs(art["dc_full"], local, visitante, bool(neutral_match))
    lh, la = art["dc_full"]["lambdas"](local, visitante, bool(neutral_match))
    res["goles"]["Dixon-Coles"] = (round(float(lh), 3), round(float(la), 3))

    res["metodos"]["ENSEMBLE"] = np.mean(list(res["metodos"].values()), axis=0)

    # marcadores mas probables (Dixon-Coles)
    M = dc_score_matrix(art["dc_full"], local, visitante, bool(neutral_match), 6)
    marc = [(i, j, M[i, j]) for i in range(7) for j in range(7)]
    marc.sort(key=lambda x: -x[2])
    res["marcadores"] = pd.DataFrame(
        [{"Local": i, "Visitante": j, "Prob %": round(p * 100, 2)} for i, j, p in marc[:8]])
    return res


# ============================================================
# 7) SIMULADOR DEL TORNEO
# ============================================================
def reconstruir_grupos(art):
    df, por_jugar = art["df"], art["por_jugar"]
    wc_j = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")]
    wc = pd.concat([wc_j, por_jugar], ignore_index=True)
    adj = collections.defaultdict(set)
    for _, r in wc.iterrows():
        adj[r["home_team"]].add(r["away_team"]); adj[r["away_team"]].add(r["home_team"])
    teams = sorted(pd.unique(pd.concat([wc["home_team"], wc["away_team"]])))
    seen, groups, gid = set(), {}, 0
    for t in teams:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack.extend(adj[x] - seen)
        groups[chr(65 + gid)] = sorted(comp); gid += 1
    fixtures = []
    for _, r in wc.iterrows():
        gh = None if pd.isna(r["home_score"]) else int(r["home_score"])
        ga = None if pd.isna(r["away_score"]) else int(r["away_score"])
        fixtures.append((r["home_team"], r["away_team"], bool(r["neutral"]), gh, ga))
    return groups, fixtures, teams


# ----------------------------------------------------------
# BRACKET OFICIAL FIFA Mundial 2026 (partidos 73-104)
# Fuente: estructura oficial del cuadro de eliminatorias.
# Cada partido R32 enfrenta ganador/subcampeon de grupo, y los 8
# mejores terceros ocupan los slots "3RD" segun la tabla de elegibilidad.
# ----------------------------------------------------------
_R32 = {
    73: ("RU_A", "RU_B"), 74: ("W_E", "3RD"), 75: ("W_F", "RU_C"),
    76: ("W_C", "RU_F"), 77: ("W_I", "3RD"), 78: ("RU_E", "RU_I"),
    79: ("W_A", "3RD"), 80: ("W_L", "3RD"), 81: ("W_D", "3RD"),
    82: ("W_G", "3RD"), 83: ("RU_K", "RU_L"), 84: ("W_H", "RU_J"),
    85: ("W_B", "3RD"), 86: ("W_J", "RU_H"), 87: ("W_K", "3RD"),
    88: ("RU_D", "RU_G"),
}
# Grupos elegibles para el tercero de cada slot (tabla FIFA)
_THIRD_ELIG = {
    74: set("ABCDF"), 77: set("CDFGH"), 79: set("CEFHI"), 80: set("EHIJK"),
    81: set("BEFIJ"), 82: set("AEHIJ"), 85: set("EFGIJ"), 87: set("DEIJL"),
}
_R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
        93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
_QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
_SF = {101: (97, 98), 102: (99, 100)}
_FINAL = (101, 102)

_STAGE = {"GROUP": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "FINAL": 5, "CHAMP": 6}


def _match_thirds(qual_groups):
    """Asigna los 8 grupos con tercero clasificado a los 8 slots '3RD'
    respetando la elegibilidad (emparejamiento bipartito, algoritmo de Kuhn).
    Verificado: existe emparejamiento perfecto para las 495 combinaciones."""
    matchR = {}  # grupo -> slot

    def assign(slot, seen):
        for g in _THIRD_ELIG[slot]:
            if g in qual_groups and g not in seen:
                seen.add(g)
                if g not in matchR or assign(matchR[g], seen):
                    matchR[g] = slot
                    return True
        return False

    for s in _THIRD_ELIG:
        assign(s, set())
    slot2group = {s: g for g, s in matchR.items()}
    # fallback defensivo (no deberia ocurrir)
    libres_s = [s for s in _THIRD_ELIG if s not in slot2group]
    libres_g = [g for g in qual_groups if g not in matchR]
    for s, g in zip(libres_s, libres_g):
        slot2group[s] = g
    return slot2group


def simulate_tournament(art, n_sim=10000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    groups, fixtures, teams = reconstruir_grupos(art)
    team2group = {t: g for g, ts in groups.items() for t in ts}
    lambdas = art["dc_full"]["lambdas"]

    def sim_score(h, a, neu):
        lh, la = lambdas(h, a, neu)
        return rng.poisson(lh), rng.poisson(la)

    def ko_winner(h, a):
        gh, ga = sim_score(h, a, True)
        if gh > ga:
            return h
        if ga > gh:
            return a
        lh, la = lambdas(h, a, True)
        return h if rng.random() < lh / (lh + la) else a

    def standings(gt, results):
        st = {t: [0, 0, 0] for t in gt}  # pts, gf, gc
        for h, a, gh, ga in results:
            st[h][1] += gh; st[h][2] += ga; st[a][1] += ga; st[a][2] += gh
            if gh > ga:
                st[h][0] += 3
            elif ga > gh:
                st[a][0] += 3
            else:
                st[h][0] += 1; st[a][0] += 1
        rows = [(t, st[t][0], st[t][1] - st[t][2], st[t][1], rng.random()) for t in gt]
        rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
        return rows

    cont = {t: collections.Counter() for t in teams}
    for _ in range(n_sim):
        gres = {g: [] for g in groups}
        for h, a, neu, gh, ga in fixtures:
            if gh is None:
                gh, ga = sim_score(h, a, neu)
            gres[team2group[h]].append((h, a, gh, ga))

        W, RU, thirds = {}, {}, []
        for g in groups:
            o = standings(groups[g], gres[g])
            W[g], RU[g] = o[0][0], o[1][0]
            thirds.append((g, o[2][0], o[2][1], o[2][2], o[2][3], o[2][4]))
        thirds.sort(key=lambda x: (-x[2], -x[3], -x[4], x[5]))
        third_team = {x[0]: x[1] for x in thirds[:8]}
        slot2group = _match_thirds(set(third_team))

        def team_of(tok, mnum):
            if tok == "3RD":
                return third_team[slot2group[mnum]]
            kind, g = tok.split("_")
            return W[g] if kind == "W" else RU[g]

        reached = {t: 0 for t in teams}
        win = {}
        for mnum, (t1, t2) in _R32.items():
            a1, a2 = team_of(t1, mnum), team_of(t2, mnum)
            reached[a1] = max(reached[a1], _STAGE["R32"])
            reached[a2] = max(reached[a2], _STAGE["R32"])
            w = ko_winner(a1, a2); win[mnum] = w
            reached[w] = _STAGE["R16"]
        for mnum, (m1, m2) in _R16.items():
            w = ko_winner(win[m1], win[m2]); win[mnum] = w
            reached[w] = _STAGE["QF"]
        for mnum, (m1, m2) in _QF.items():
            w = ko_winner(win[m1], win[m2]); win[mnum] = w
            reached[w] = _STAGE["SF"]
        for mnum, (m1, m2) in _SF.items():
            w = ko_winner(win[m1], win[m2]); win[mnum] = w
            reached[w] = _STAGE["FINAL"]
        champ = ko_winner(win[_FINAL[0]], win[_FINAL[1]])
        reached[champ] = _STAGE["CHAMP"]

        for t in teams:
            st = reached[t]
            for nombre, rank in _STAGE.items():
                if rank >= 1 and st >= rank:
                    cont[t][nombre] += 1

    filas = []
    for t in teams:
        c = cont[t]
        filas.append({"PAIS": t, "GRUPO": team2group[t], "ELO": round(art["elo"].get(t, 1500)),
                      "P_Clasifica": 100 * c["R32"] / n_sim, "P_Octavos": 100 * c["R16"] / n_sim,
                      "P_Cuartos": 100 * c["QF"] / n_sim, "P_Semis": 100 * c["SF"] / n_sim,
                      "P_Final": 100 * c["FINAL"] / n_sim, "P_CAMPEON": 100 * c["CHAMP"] / n_sim})
    prob = (pd.DataFrame(filas).sort_values("P_CAMPEON", ascending=False)
            .reset_index(drop=True).round(2))
    return prob, groups
