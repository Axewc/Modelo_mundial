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

import os
import collections
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
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
    """Calibra un modelo ya entrenado de forma compatible entre versiones.
    Intenta FrozenEstimator (sklearn >=1.6, recomendado) y, si falla por
    cualquier motivo (import o validacion), cae a cv='prefit' (sklearn <1.6).
    El .fit va DENTRO del try para capturar tambien errores de validacion."""
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic")
        cal.fit(X_val, y_val)
        return cal
    except Exception:
        cal = CalibratedClassifierCV(modelo, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        return cal

from pathlib import Path as _Path
_RAIZ = _Path(__file__).resolve().parents[1]
CSV_DEFAULT = str(_RAIZ / "results_actualizado_mundial_2026.csv")
# Estructura OFICIAL FIFA 2026 (grupos + cuadro) provista como datasets.
MATCHES_CSV_DEFAULT = "matches.csv"
TEAMS_CSV_DEFAULT = "teams.csv"
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
def load_data(path=CSV_DEFAULT, extra=None):
    df_raw = pd.read_csv(path)
    # Canoniza nombres de seleccion (une grafias inconsistentes del propio
    # dataset, p.ej. "Congo DR" y "DR Congo" -> una sola seleccion).
    try:
        from mundial import nombres
        for col in ("home_team", "away_team"):
            df_raw[col] = df_raw[col].map(nombres.normaliza)
    except Exception:
        pass
    if extra is not None and len(extra) > 0:
        ex = extra.copy()
        for c in df_raw.columns:
            if c not in ex.columns:
                ex[c] = np.nan
        df_raw = pd.concat([df_raw, ex[df_raw.columns]], ignore_index=True)
        # Dedup SOLO de las claves añadidas: el partido nuevo (al final) gana
        # ante un fixture ya existente, sin tocar duplicados preexistentes.
        keys = ["date", "home_team", "away_team"]
        ex_keys = set(map(tuple, ex[keys].astype(str).itertuples(index=False)))
        en_extra = pd.Series(
            [t in ex_keys for t in df_raw[keys].astype(str).itertuples(index=False)],
            index=df_raw.index)
        dup = df_raw.duplicated(subset=keys, keep="last")
        df_raw = df_raw[~(dup & en_extra)].reset_index(drop=True)
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
def compute_elo(df_full, anio_inicio=None):
    """anio_inicio: año desde el que se recorta el frame devuelto (default:
    ANIO_INICIO). Permite evaluaciones point-in-time (p.ej. backtest de
    Mundiales pasados) usando la MISMA receta con la ventana desplazada."""
    if anio_inicio is None:
        anio_inicio = ANIO_INICIO
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
            pc = 0.60
        elif "qualification" in str(t).lower():
            pc = 0.90
        elif t in COMP_FUERTE:
            pc = 1.10
        else:
            pc = 0.80

        k = 24 * pr * pc * mov
        elo[h] += np.clip(k * (sh - exp_home), -32, 32)
        elo[a] += np.clip(k * (sa - (1 - exp_home)), -32, 32)

    d["elo_home"] = elo_home
    d["elo_away"] = elo_away
    d["elo_diff"] = d["elo_home"] - d["elo_away"]

    df = d[d["date"] >= f"{anio_inicio}-01-01"].reset_index(drop=True).copy()
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
    pc = np.where(t.eq("Friendly"), 0.60,
         np.where(t.str.lower().str.contains("qualification"), 0.90,
         np.where(t.isin(COMP_FUERTE), 1.10, 0.80)))
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
        # cota de seguridad: ninguna seleccion espera <0.05 ni >6 goles
        return float(np.clip(lh, 0.05, 6.0)), float(np.clip(la, 0.05, 6.0))

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


def prob_avance(p1x2, lh, la, shootout=0.5, prorroga=True, max_goals=8):
    """Eliminatoria a partido único: convierte el 1X2 [away, draw, home] en
    P(avanza). El empate de los 90' se rompe con PRÓRROGA (mini-Poisson a 1/3
    de los goles esperados) y, si persiste, PENALES.
      shootout = prob de que el LOCAL gane la tanda (0.5 = lotería).
    Devuelve [avanza_away, avanza_home] (mismo orden que el 1X2)."""
    p_away, p_draw, p_home = (float(x) for x in p1x2)
    if prorroga and lh > 0 and la > 0:
        kh = poisson.pmf(np.arange(max_goals + 1), lh / 3.0)
        ka = poisson.pmf(np.arange(max_goals + 1), la / 3.0)
        M = np.outer(kh, ka)
        et_h = float(np.tril(M, -1).sum())
        et_d = float(np.trace(M))
        et_a = float(np.triu(M, 1).sum())
        s = et_h + et_d + et_a
        et_h, et_d, et_a = et_h / s, et_d / s, et_a / s
        local_si_empate = et_h + et_d * shootout       # gana prórroga o penales
        visit_si_empate = et_a + et_d * (1 - shootout)
    else:
        local_si_empate, visit_si_empate = shootout, 1 - shootout
    av_home = p_home + p_draw * local_si_empate
    av_away = p_away + p_draw * visit_si_empate
    t = av_home + av_away
    return np.array([av_away / t, av_home / t])


# ============================================================
# 4c) BINOMIAL NEGATIVA (cola más gruesa que Poisson)
# ============================================================
# Misma media lambda del GLM de Dixon-Coles, pero distribución de conteo con
# sobredispersion (var = mu + mu^2/r). r grande -> Poisson; r chico -> mas cola
# (mas goleadas, y tambien mas 0-0). Se conserva la correccion rho.
def _nb_pmf(ks, mu, r):
    mu = max(float(mu), 1e-6)
    return nbinom.pmf(ks, r, r / (r + mu))


def nb_score_matrix(pred, home, away, neutral, r, max_goals=10):
    lh, la = pred["lambdas"](home, away, neutral)
    rho = pred["rho"]
    ks = np.arange(max_goals + 1)
    h = _nb_pmf(ks, lh, r)
    a = _nb_pmf(ks, la, r)
    M = np.outer(h, a)
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


def nb_result_probs(pred, home, away, neutral, r, max_goals=10):
    M = nb_score_matrix(pred, home, away, neutral, r, max_goals)
    return np.array([np.triu(M, 1).sum(), np.trace(M), np.tril(M, -1).sum()])


def select_size_nb(pred, d_val, sizes=(2, 3, 4, 6, 10, 20, 50)):
    """Elige el parámetro de dispersión r minimizando el log-loss 1X2 sobre
    un periodo de validación. `pred` es un Dixon-Coles ya ajustado (la media
    no cambia con r). r alto = casi Poisson."""
    y = d_val.apply(
        lambda x: 2 if x["home_score"] > x["away_score"]
        else (0 if x["home_score"] < x["away_score"] else 1), axis=1).values
    best_r, best_ll = sizes[len(sizes) // 2], np.inf
    for r in sizes:
        try:
            P = np.array([nb_result_probs(pred, h, a, bool(n), r) for h, a, n in
                          zip(d_val["home_team"], d_val["away_team"],
                              d_val["neutral_match"])])
        except Exception:
            continue
        ll = log_loss(y, np.clip(P, 1e-15, 1), labels=CLASSES)
        if ll < best_ll:
            best_ll, best_r = ll, r
    return best_r


# --- Seleccion de la regularizacion (alpha) del modelo de goles DC ---
def _dc_lambdas_df(pred, d):
    atk, dfn = pred["atk"], pred["dfn"]
    c, g = pred["c"], pred["gamma"]
    ah = d["home_team"].map(atk).fillna(0.0).values
    dh = d["home_team"].map(dfn).fillna(0.0).values
    aa = d["away_team"].map(atk).fillna(0.0).values
    da = d["away_team"].map(dfn).fillna(0.0).values
    hb = (1 - d["neutral_match"].values)
    lh = np.clip(np.exp(c + ah + da + g * hb), 0.05, 6.0)
    la = np.clip(np.exp(c + aa + dh), 0.05, 6.0)
    return lh, la


def _poisson_deviance(y, lam):
    y = np.asarray(y, float)
    lam = np.clip(np.asarray(lam, float), 1e-9, None)
    # usar 1.0 donde y==0 evita log(0); esos terminos se anulan igual (y=0)
    safe = np.where(y > 0, y, 1.0)
    term = np.where(y > 0, y * np.log(safe / lam), 0.0)
    return 2.0 * np.sum(term - (y - lam))


def _dc_deviance(pred, d):
    lh, la = _dc_lambdas_df(pred, d)
    return (_poisson_deviance(d["home_score"].values, lh)
            + _poisson_deviance(d["away_score"].values, la))


def select_alpha_dc(d_tr, d_val, alphas=(0.0003, 0.001, 0.003, 0.01, 0.05, 0.5)):
    """Elige la regularizacion del modelo de goles minimizando la deviance
    de Poisson sobre un periodo de validacion (en vez de fijarla a ojo).

    Grid extendido hacia abajo tras validarlo en 5 cortes temporales
    (2021-2025): alpha<=0.001 gano unanimemente por deviance Y por RPS
    (RPS out-of-sample ~0.175 vs ~0.20 del grid anterior, que se quedaba
    corto en 0.01). El piso 0.0003 conserva algo de regularizacion para
    estabilidad; bajar mas solo anade ~0.003 de RPS."""
    best_a, best_dev = 0.01, np.inf
    for a in alphas:
        try:
            pred = fit_dc(d_tr, alpha=a)
            dev = _dc_deviance(pred, d_val)
        except Exception:
            continue
        if np.isfinite(dev) and dev < best_dev:
            best_dev, best_a = dev, a
    return best_a


# ============================================================
# 4b) VALIDACION TEMPORAL (walk-forward, ventana expansiva)
# ============================================================
def temporal_validation(df, dm, n_folds=5, min_train_frac=0.5, include_xgb=True):
    """Evaluacion walk-forward: en vez de un unico split 80/20, entrena con
    una ventana que se va expandiendo en el tiempo y evalua en el bloque
    siguiente (varios folds). Asi se mide la robustez fuera de muestra de
    forma realista (siempre se predice el futuro con el pasado) y se detecta
    overfit. `df` y `dm` deben estar alineados por posicion (mismo orden).

    Devuelve un dict con:
      - "folds":   metricas por (modelo, fold)
      - "resumen": media y desviacion de cada metrica por modelo
    """
    n = len(dm)
    start = max(int(n * min_train_frac), 60)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    fb = ["elo_diff", "home_advantage", "neutral_match"]
    filas = []

    for i in range(n_folds):
        cut, end = int(bounds[i]), int(bounds[i + 1])
        if cut < 60 or end - cut < 15:
            continue
        tr_dm, te_dm = dm.iloc[:cut], dm.iloc[cut:end]
        tr_df, te_df = df.iloc[:cut], df.iloc[cut:end]
        y_true = te_dm["resultado"].values
        yr_tr = tr_dm["resultado"]
        meta = {"fold": i + 1, "n_test": int(len(te_dm))}

        # Baseline ELO (regresion logistica sobre elo_diff)
        try:
            blr = LogisticRegression(max_iter=1000).fit(tr_dm[fb], yr_tr)
            pb = blr.predict_proba(te_dm[fb])[:, np.argsort(blr.classes_)]
            filas.append({**meta, **_evaluar("Baseline ELO", y_true, pb)})
        except Exception:
            pass

        # Poisson (Monte Carlo sobre lambdas predichas)
        try:
            sc = StandardScaler().fit(tr_dm[FEATURES])
            Xtr_s, Xte_s = sc.transform(tr_dm[FEATURES]), sc.transform(te_dm[FEATURES])
            mh = PoissonRegressor(alpha=0.1, max_iter=500).fit(Xtr_s, tr_dm["home_score"])
            ma = PoissonRegressor(alpha=0.1, max_iter=500).fit(Xtr_s, tr_dm["away_score"])
            ph, pa = mh.predict(Xte_s), ma.predict(Xte_s)
            pp = np.array([poisson_result_probs(lh, la) for lh, la in zip(ph, pa)])
            filas.append({**meta, **_evaluar("Poisson (MC)", y_true, pp)})
        except Exception:
            pass

        # Dixon-Coles (alpha elegido por validacion interna del fold)
        try:
            a_cut = int(cut * 0.85)
            alpha_dc = (select_alpha_dc(tr_df.iloc[:a_cut], tr_df.iloc[a_cut:])
                        if a_cut > 40 else 0.05)
            dcp = fit_dc(tr_df, alpha=alpha_dc)
            pdc = np.array([dc_result_probs(dcp, h, a, bool(nn)) for h, a, nn in
                            zip(te_df["home_team"], te_df["away_team"],
                                te_df["neutral_match"])])
            filas.append({**meta, **_evaluar("Dixon-Coles", y_true, pdc)})
        except Exception:
            pass

        # XGBoost (opcional; early stopping sobre la cola del train del fold)
        if include_xgb and HAS_XGB:
            try:
                vcut = int(cut * 0.85)
                mx = XGBClassifier(
                    n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=3, gamma=0.2,
                    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                    early_stopping_rounds=50, random_state=RANDOM_STATE)
                mx.fit(tr_dm[FEATURES].iloc[:vcut], yr_tr.iloc[:vcut],
                       eval_set=[(tr_dm[FEATURES].iloc[vcut:], yr_tr.iloc[vcut:])],
                       verbose=False)
                filas.append({**meta, **_evaluar("XGBoost", y_true,
                                                 mx.predict_proba(te_dm[FEATURES]))})
            except Exception:
                pass

    folds = pd.DataFrame(filas)
    if folds.empty:
        return {"folds": folds, "resumen": pd.DataFrame()}

    g = folds.groupby("Modelo")
    resumen = pd.DataFrame({
        "Modelo": list(g.groups.keys()),
        "Folds": g["fold"].count().values,
        "LogLoss": g["LogLoss"].mean().values,
        "LogLoss_sd": g["LogLoss"].std(ddof=0).values,
        "RPS": g["RPS"].mean().values,
        "RPS_sd": g["RPS"].std(ddof=0).values,
        "Brier": g["Brier"].mean().values,
        "Accuracy": g["Accuracy"].mean().values,
    }).sort_values("RPS").reset_index(drop=True).round(4)
    return {"folds": folds.round(4), "resumen": resumen}


# ============================================================
# 5) PIPELINE COMPLETO
# ============================================================
def _brier(y, proba):
    y = np.asarray(y)
    Y = np.zeros((len(y), 3))
    for i, c in enumerate(CLASSES):
        Y[:, i] = (y == c)
    return np.mean(np.sum((proba - Y) ** 2, axis=1))


def _rps(y, proba):
    """Ranked Probability Score: metrica estandar para pronosticos 1X2
    (penaliza segun la distancia ORDINAL entre la prob acumulada predicha
    y la real). Orden: [gana visitante, empate, gana local]. Menor = mejor."""
    y = np.asarray(y)
    Y = np.zeros((len(y), 3))
    for i, c in enumerate(CLASSES):
        Y[:, i] = (y == c)
    cdf_p = np.cumsum(proba, axis=1)
    cdf_y = np.cumsum(Y, axis=1)
    return float(np.mean(np.sum((cdf_p - cdf_y) ** 2, axis=1) / (3 - 1)))


def _evaluar(nombre, y, proba):
    proba = np.clip(np.asarray(proba, float), 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[np.argmax(proba, axis=1)]
    return {"Modelo": nombre, "LogLoss": log_loss(y, proba, labels=CLASSES),
            "Brier": _brier(y, proba), "RPS": _rps(y, proba),
            "Accuracy": accuracy_score(y, pred)}


def run_pipeline(path=CSV_DEFAULT, extra=None):
    df_full, por_jugar = load_data(path, extra=extra)
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

    # dixon-coles: elegir regularizacion (alpha) por validacion temporal
    try:
        a_cut = int(split * 0.85)
        alpha_dc = select_alpha_dc(df.iloc[:a_cut], df.iloc[a_cut:split])
    except Exception:
        alpha_dc = 0.05
    dc_eval = fit_dc(df.iloc[:split], alpha=alpha_dc)
    pdc = np.array([dc_result_probs(dc_eval, h, a, bool(n)) for h, a, n in
                    zip(df.iloc[split:]["home_team"], df.iloc[split:]["away_team"],
                        df.iloc[split:]["neutral_match"])])
    evals.append(_evaluar("Dixon-Coles", y_true, pdc))

    # neg-binomial: misma media DC, dispersion r elegida por validacion
    try:
        r_nb = select_size_nb(fit_dc(df.iloc[:a_cut], alpha=alpha_dc),
                              df.iloc[a_cut:split])
    except Exception:
        r_nb = 10
    pnb = np.array([nb_result_probs(dc_eval, h, a, bool(n), r_nb) for h, a, n in
                    zip(df.iloc[split:]["home_team"], df.iloc[split:]["away_team"],
                        df.iloc[split:]["neutral_match"])])
    evals.append(_evaluar("Neg-Binomial", y_true, pnb))

    dc_full = fit_dc(df, alpha=alpha_dc)

    tabla_eval = (pd.DataFrame(evals).sort_values("LogLoss")
                  .reset_index(drop=True).round(4))

    # NOTA: la validacion temporal walk-forward (temporal_validation) es
    # costosa (reentrena varios modelos por fold). NO se ejecuta aqui para
    # no bloquear el primer arranque de la app; se calcula bajo demanda
    # usando los `df` y `dm` que devolvemos en el dict (ver app.py).

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
        "df": df, "dm": dm, "por_jugar": por_jugar, "elo": elo, "ranking": ranking,
        "scaler": scaler, "m_home": m_home, "m_away": m_away,
        "m_xgb": m_xgb, "m_xgb_cal": m_xgb_cal, "dc_full": dc_full,
        "h2h": h2h, "equipos_actuales": equipos_actuales,
        "tabla_eval": tabla_eval, "has_xgb": HAS_XGB,
        "alpha_dc": alpha_dc, "dc_rho": dc_full["rho"], "dc_gamma": dc_full["gamma"],
        "r_nb": r_nb,
        "mae_home": float(np.mean(np.abs(pred_home - test["home_score"].values))),
        "mae_away": float(np.mean(np.abs(pred_away - test["away_score"].values))),
    }


# ============================================================
# 5b) ¿"Analisis de forma" vs el resto? (decidir por RPS, no por autoridad)
# ============================================================
def evaluar_forma(art, ranking=None, test_frac=0.20, max_eval=None):
    """Evalua el modelo de 'Analisis de forma' (analisis_forma.py) en el MISMO
    set de test que el resto, POINT-IN-TIME: para cada partido, solo usa datos
    anteriores. Devuelve la fila de metricas (RPS/LogLoss/Brier/Accuracy) para
    poder compararlo directamente con Dixon-Coles / XGBoost.

    NOTA: el ranking FIFA y el ELO que recibe la forma son los ACTUALES (no se
    recalculan hacia atras), un leak menor que, si acaso, FAVORECE a la forma.
    Es decir, si Dixon-Coles aun le gana, la conclusion es conservadora.
    """
    import analisis_forma as af

    df = art["df"].reset_index(drop=True)
    elo = art["elo"]
    n = len(df)
    split = int(n * (1 - test_frac))
    idxs = list(range(split, n))
    if max_eval:
        idxs = idxs[-int(max_eval):]

    P, y = [], []
    for pos in idxs:
        row = df.iloc[pos]
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        df_upto = df.iloc[:pos]
        if len(df_upto) < 50:
            continue
        try:
            r = af.predecir_partido_forma(
                df_upto, elo, ranking, row["home_team"], row["away_team"],
                bool(row["neutral_match"]))
        except Exception:
            continue
        pr = r["prob"]
        P.append([pr["visitante"], pr["empate"], pr["local"]])  # [away,draw,home]
        if row["home_score"] > row["away_score"]:
            y.append(2)
        elif row["home_score"] < row["away_score"]:
            y.append(0)
        else:
            y.append(1)

    if not P:
        return None
    return {**_evaluar("Analisis de forma", np.array(y), np.array(P)),
            "n_test": len(y)}


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

    # Ensemble SOLO con los metodos solidos (excluye el XGB sin calibrar)
    keys = [k for k in ["Dixon-Coles", "XGBoost (calibrado)", "Poisson + MC"]
            if k in res["metodos"]]
    res["metodos"]["ENSEMBLE"] = np.mean([res["metodos"][k] for k in keys], axis=0)

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
GRUPO_TAM = 4  # Mundial 2026: 12 grupos de 4 selecciones


def cargar_estructura_oficial(matches_path=MATCHES_CSV_DEFAULT,
                              teams_path=TEAMS_CSV_DEFAULT):
    """Lee la estructura OFICIAL FIFA 2026 de `teams.csv` + `matches.csv`.

    Devuelve un dict con:
      - `grupo_de`: {nombre_canónico: letra_grupo} para las 42 selecciones
        reales (los 6 placeholders de repesca se resuelven luego por el grafo).
      - `r32`: {match_number: (tok1, tok2)} con tokens '1E' (1º de E),
        '2C' (2º de C) o '3ABCDF' (mejor 3º de uno de esos grupos).
      - `third_elig`: {match_number: set('ABCDF')} para los 8 slots de terceros.

    Devuelve None si falta algún archivo o el formato no cuadra (la app cae
    entonces al cuadro cableado heredado)."""
    if not (os.path.exists(matches_path) and os.path.exists(teams_path)):
        return None
    try:
        import nombres as _nombres
        tm = pd.read_csv(teams_path)
        mt = pd.read_csv(matches_path)
        grupo_de = {_nombres.normaliza(str(r["team_name"])): str(r["group_letter"])
                    for _, r in tm.iterrows()
                    if not bool(r["is_placeholder"])}
        r32, third_elig = {}, {}
        for _, r in mt[mt["stage_id"] == 2].iterrows():
            lab = str(r["match_label"])
            t1, t2 = (s.strip() for s in lab.split(" vs "))
            mnum = int(r["match_number"])
            r32[mnum] = (t1, t2)
            for tok in (t1, t2):
                if tok[:1] == "3":
                    third_elig[mnum] = set(tok[1:])
        if len(r32) != 16 or len(third_elig) != 8:
            return None
        return {"grupo_de": grupo_de, "r32": r32, "third_elig": third_elig}
    except Exception:
        return None


def _reetiquetar_oficial(groups, grupo_de):
    """Cambia las etiquetas alfabéticas por las letras OFICIALES de cada grupo,
    deducidas de sus miembros reales (los placeholders heredan la del grupo).
    Si algo no cuadra (letra ausente o repetida), conserva las etiquetas
    alfabéticas para no romper el cuadro."""
    try:
        import nombres as _nombres
    except Exception:
        return groups
    nuevo = {}
    for ts in groups.values():
        votos = collections.Counter(grupo_de.get(_nombres.normaliza(t)) for t in ts)
        votos.pop(None, None)
        letra = votos.most_common(1)[0][0] if votos else None
        if letra is None or letra in nuevo:
            return groups
        nuevo[letra] = ts
    return {g: nuevo[g] for g in sorted(nuevo)}


def reconstruir_grupos(art):
    """Reconstruye los 12 grupos a partir del calendario del Mundial.

    La fase de grupos es un round-robin dentro de cada grupo de 4; los partidos
    de ELIMINATORIA cruzan grupos. Si esas aristas se incluyeran (como hacía la
    versión por componentes conexas), al empezar los dieciseisavos los 12 grupos
    se fundirían en pocos bloques gigantes y el cuadro (`_R32`) lanzaría
    `KeyError`. Para evitarlo usamos union-find en orden cronológico (los
    partidos de grupo preceden a los de eliminatoria) y RECHAZAMOS cualquier
    arista que haría crecer un grupo por encima de `GRUPO_TAM`: justamente los
    cruces de eliminatoria. Devuelve además SOLO los fixtures intra-grupo, para
    que la tabla de cada grupo no mezcle partidos de eliminatoria."""
    df, por_jugar = art["df"], art["por_jugar"]
    wc_j = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")]
    wc = pd.concat([wc_j, por_jugar], ignore_index=True)
    teams = sorted(pd.unique(pd.concat([wc["home_team"], wc["away_team"]])))

    parent = {t: t for t in teams}
    size = {t: 1 for t in teams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # compresión de caminos
            x = parent[x]
        return x

    for _, r in wc.sort_values("date").iterrows():
        ra, rb = find(r["home_team"]), find(r["away_team"])
        if ra == rb:
            continue
        if size[ra] + size[rb] > GRUPO_TAM:
            continue  # cruza grupos (eliminatoria) -> no une
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    comps = collections.defaultdict(list)
    for t in teams:
        comps[find(t)].append(t)
    # Etiqueta A, B, C… de forma estable (por el equipo alfabéticamente menor).
    groups = {chr(65 + i): sorted(comps[root])
              for i, root in enumerate(sorted(comps, key=lambda r: min(comps[r])))}
    # Si hay estructura oficial, reemplaza por las letras de grupo reales (FIFA).
    est = cargar_estructura_oficial()
    if est:
        groups = _reetiquetar_oficial(groups, est["grupo_de"])
    t2g = {t: g for g, ts in groups.items() for t in ts}

    fixtures = []
    for _, r in wc.iterrows():
        h, a = r["home_team"], r["away_team"]
        if t2g.get(h) != t2g.get(a):
            continue  # solo partidos de fase de grupos (intra-grupo)
        gh = None if pd.isna(r["home_score"]) else int(r["home_score"])
        ga = None if pd.isna(r["away_score"]) else int(r["away_score"])
        fixtures.append((h, a, bool(r["neutral"]), gh, ga))
    return groups, fixtures, teams


# ----------------------------------------------------------
# BRACKET — Mundial 2026 (partidos 73-104)
# El R32 (qué grupo va a cada slot, y la elegibilidad de los terceros) se TOMA
# del dataset oficial `matches.csv` vía cargar_estructura_oficial(). Lo de abajo
# es el FALLBACK heredado por si esos archivos no están: ¡ojo! este cableado no
# coincide con el cuadro oficial 2026 (estaba descuadrado), por eso se prefiere
# siempre la estructura oficial. El árbol R16→Final sí es estable y se reutiliza
# en ambos casos (las llaves 73-88 coinciden con los match_number oficiales).
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


def _r32_a_tokens(est):
    """Devuelve (r32, third_elig) en tokens oficiales ('1E','2C','3ABCDF').
    Usa la estructura oficial si está; si no, convierte el cuadro heredado."""
    if est:
        return est["r32"], est["third_elig"]

    def conv(tok, mnum):
        if tok == "3RD":
            return "3" + "".join(sorted(_THIRD_ELIG[mnum]))
        kind, g = tok.split("_")
        return ("1" if kind == "W" else "2") + g

    r32 = {m: (conv(a, m), conv(b, m)) for m, (a, b) in _R32.items()}
    return r32, {m: set(v) for m, v in _THIRD_ELIG.items()}


def _match_thirds(qual_groups, third_elig):
    """Asigna los 8 grupos con tercero clasificado a los 8 slots '3RD'
    respetando la elegibilidad (emparejamiento bipartito, algoritmo de Kuhn).
    `third_elig`: {match_number: set(grupos_elegibles)}."""
    matchR = {}  # grupo -> slot

    def assign(slot, seen):
        for g in third_elig[slot]:
            if g in qual_groups and g not in seen:
                seen.add(g)
                if g not in matchR or assign(matchR[g], seen):
                    matchR[g] = slot
                    return True
        return False

    for s in third_elig:
        assign(s, set())
    slot2group = {s: g for g, s in matchR.items()}
    # fallback defensivo (no deberia ocurrir)
    libres_s = [s for s in third_elig if s not in slot2group]
    libres_g = [g for g in qual_groups if g not in matchR]
    for s, g in zip(libres_s, libres_g):
        slot2group[s] = g
    return slot2group


def simulate_tournament(art, n_sim=10000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    groups, fixtures, teams = reconstruir_grupos(art)
    team2group = {t: g for g, ts in groups.items() for t in ts}
    # Cuadro de eliminatoria: oficial (matches.csv) o, si no está, el heredado.
    r32, third_elig = _r32_a_tokens(cargar_estructura_oficial())
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
        slot2group = _match_thirds(set(third_team), third_elig)

        def team_of(tok, mnum):
            pos, g = tok[0], tok[1:]
            if pos == "1":
                return W[g]
            if pos == "2":
                return RU[g]
            return third_team[slot2group[mnum]]  # pos == "3"

        reached = {t: 0 for t in teams}
        win = {}
        for mnum, (t1, t2) in r32.items():
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
