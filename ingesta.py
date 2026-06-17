"""
ingesta.py — Ingesta de "partidos del día" y contraste modelo vs realidad
=========================================================================
Pensado para usarse durante el torneo: cada jornada se traen/añaden los
partidos del día, se guarda la predicción del modelo, y al final del día se
registran los resultados reales para:

  1. CONTRASTAR lo modelado contra lo que pasó (acierto, Brier, log-loss).
  2. ENRIQUECER el dataset y reentrenar con esos partidos nuevos.

Persistencia: en Streamlit Community Cloud el disco es efímero, así que el
estado vive en la sesión y se exporta con un botón de descarga (el usuario
re-sube el CSV al repo). El traer automático usa la API pública de ESPN
(sin clave); si no hay red o cambia el formato, se cae a ingesta manual.
"""
import numpy as np
import pandas as pd

# Columnas canónicas del CSV de resultados.
COLS = ["date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral"]

# Normalización de nombres de fuentes externas -> nombres del CSV de resultados.
NOMBRES_EXT = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Korea DPR": "North Korea", "IR Iran": "Iran", "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde", "Türkiye": "Turkey", "Turkiye": "Turkey",
    "China": "China PR", "Chinese Taipei": "Taiwan",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
}


def normaliza_equipo(n):
    return NOMBRES_EXT.get(str(n).strip(), str(n).strip())


def resultado_clase(home_score, away_score):
    """Codifica el resultado: 2 gana local, 1 empate, 0 gana visitante
    (mismo orden [away, draw, home] que las probabilidades del modelo)."""
    if pd.isna(home_score) or pd.isna(away_score):
        return np.nan
    if home_score > away_score:
        return 2
    if home_score < away_score:
        return 0
    return 1


# ============================================================
# Traer partidos del día (automático, ESPN; sin clave)
# ============================================================
def parse_espn_scoreboard(j, tournament="FIFA World Cup", neutral=True):
    """Convierte el JSON del scoreboard de ESPN en filas estándar.
    Aislado del request para poder testearlo sin red."""
    filas = []
    for ev in j.get("events", []):
        comps = (ev.get("competitions") or [{}])[0]
        teams = comps.get("competitors", [])
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        completed = (((comps.get("status") or {}).get("type") or {})
                     .get("completed", False))

        def nombre(t):
            tm = t.get("team") or {}
            return normaliza_equipo(tm.get("displayName") or tm.get("name") or "")

        def score(t):
            try:
                return float(t.get("score"))
            except (TypeError, ValueError):
                return np.nan

        filas.append({
            "date": (ev.get("date") or "")[:10],
            "home_team": nombre(home), "away_team": nombre(away),
            "home_score": score(home) if completed else np.nan,
            "away_score": score(away) if completed else np.nan,
            "tournament": tournament, "city": None, "country": None,
            "neutral": neutral,
        })
    return pd.DataFrame(filas, columns=COLS)


def traer_partidos_del_dia(fecha, slug="fifa.world", tournament="FIFA World Cup"):
    """Descarga los partidos de ESPN para una fecha (YYYY-MM-DD). Requiere red.
    `slug` permite cambiar de competición (p.ej. 'fifa.worldq.concacaf')."""
    import requests
    d = pd.to_datetime(fecha).strftime("%Y%m%d")
    url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}"
           f"/scoreboard?dates={d}")
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return parse_espn_scoreboard(r.json(), tournament=tournament, neutral=True)


# ============================================================
# Contraste modelo vs realidad
# ============================================================
def evalua_contraste(pred_log):
    """A partir del registro de predicciones (con columnas p_away/p_draw/p_home
    y 'res' = clase real o NaN), calcula métricas por partido y agregadas.

    Devuelve (detalle, resumen). 'detalle' solo incluye partidos ya resueltos."""
    df = pred_log.copy()
    if "res" not in df.columns:
        return df.iloc[0:0], {}
    df = df[df["res"].notna()].copy()
    if df.empty:
        return df, {}
    P = df[["p_away", "p_draw", "p_home"]].to_numpy(dtype=float)
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y = df["res"].astype(int).to_numpy()
    pred = np.argmax(P, axis=1)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    df["acierto"] = (pred == y).astype(int)
    df["brier"] = np.sum((P - onehot) ** 2, axis=1)
    df["logloss"] = -np.log(P[np.arange(len(y)), y])
    # baseline ingenuo (33/33/33) para tener referencia
    base_ll = -np.log(1 / 3)
    resumen = {
        "n": int(len(df)),
        "accuracy": float(df["acierto"].mean()),
        "brier": float(df["brier"].mean()),
        "logloss": float(df["logloss"].mean()),
        "logloss_baseline": float(base_ll),
    }
    return df, resumen


# ============================================================
# Utilidades para el dataset
# ============================================================
def a_filas_dataset(registros):
    """Normaliza una lista de dicts a un DataFrame con las columnas del CSV."""
    if not registros:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(registros)
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS]


def merge_resultados(csv_df, nuevos_df):
    """Une el CSV original con los partidos nuevos, dando prioridad a estos
    últimos cuando coinciden (date, home_team, away_team)."""
    base = csv_df.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    nv = nuevos_df.copy()
    nv["date"] = pd.to_datetime(nv["date"]).dt.strftime("%Y-%m-%d")
    out = pd.concat([base, nv[base.columns.intersection(nv.columns)]],
                    ignore_index=True)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"],
                              keep="last").reset_index(drop=True)
    return out
