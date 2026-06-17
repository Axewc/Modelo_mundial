"""
analisis_forma.py — Análisis de forma reciente y expectativa de goles
=====================================================================
Vista transparente, complementaria a los modelos del notebook. Para un
partido A vs B construye la expectativa de goles de cada selección a partir
de sus ÚLTIMOS N PARTIDOS OFICIALES, siguiendo estos pasos (todos visibles
para el usuario en la app):

  1. Toma los últimos N partidos oficiales de cada selección (del CSV).
  2. Aplica DECAIMIENTO TEMPORAL: los partidos más nuevos pesan más.
  3. PONDERA POR RIVAL: golear/aguantar a una potencia vale más que hacerlo
     contra una selección floja (usa el ranking FIFA como medida de nivel;
     si no hay ranking, cae al ELO del modelo).
  4. ANCLA al ranking FIFA: encoge la forma (ruidosa, 12 partidos) hacia lo
     que el ranking implica, para no sobre-reaccionar a una buena/mala racha.
  5. Cruza el HISTORIAL DIRECTO (H2H): si una selección domina al rival
     históricamente, ajusta levemente la expectativa.
  6. Construye λ (goles esperados) de cada equipo y, sobre la matriz de
     marcadores, aplica la CORRECCIÓN DE RESULTADOS BAJOS (estilo
     Dixon-Coles) para destacar 0-0, 1-0, 1-1, etc.

El ranking FIFA se carga de un CSV (ver scrape_fifa_ranking.py). Si el CSV
no existe, todo funciona igual usando el ELO como ancla.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

# ---- parámetros (tunables) ----
N_PARTIDOS = 12          # últimos partidos oficiales por selección
HALF_LIFE_DIAS = 400     # vida media del decaimiento temporal de la forma
BETA_ANCLA = 0.35        # cuánto se encoge la forma hacia el ranking (0..1)
PESO_H2H = 0.12          # influencia del historial directo
RHO_BAJOS = -0.06        # corrección Dixon-Coles (negativo realza empates/bajos)
GAMMA_LOCAL = 1.10       # ventaja del local (multiplica su λ si no es neutral)
MU_LIGA = 1.35           # goles medios de referencia por equipo y partido
MAX_GOLES = 8
FIFA_CSV_DEFAULT = "fifa_ranking.csv"

# Mapa de nombres del ranking FIFA/Wikipedia -> nombres del CSV de resultados.
# Solo hace falta para los que difieren; el resto casa por nombre idéntico.
NOMBRES_FIFA_A_CSV = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "China PR": "China PR",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Czechia": "Czech Republic",
    "Cape Verde Islands": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Kosovo": "Kosovo",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei",
    "Hong Kong, China": "Hong Kong",
    "Chinese Taipei": "Taiwan",
    # nombres confirmados contra el fifa_ranking.csv real (API de FIFA)
    "Cabo Verde": "Cape Verde",
    "DPR Korea": "North Korea",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "The Gambia": "Gambia",
    "US Virgin Islands": "United States Virgin Islands",
}


# ============================================================
# Ranking FIFA
# ============================================================
def cargar_ranking_fifa(path=FIFA_CSV_DEFAULT):
    """Lee el CSV de ranking FIFA y devuelve {seleccion_csv: {rank, points}}.
    Normaliza los nombres a los del CSV de resultados. Devuelve None si el
    archivo no existe (la app cae al ELO como ancla)."""
    if not os.path.exists(path):
        return None
    try:
        r = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower(): c for c in r.columns}
    c_team = cols.get("team") or cols.get("country") or cols.get("seleccion")
    c_rank = cols.get("rank") or cols.get("ranking") or cols.get("position")
    c_pts = cols.get("points") or cols.get("puntos") or cols.get("total_points")
    if c_team is None or c_pts is None:
        return None
    out = {}
    for _, row in r.iterrows():
        nombre = str(row[c_team]).strip()
        nombre = NOMBRES_FIFA_A_CSV.get(nombre, nombre)
        try:
            pts = float(row[c_pts])
        except (ValueError, TypeError):
            continue
        rk = int(row[c_rank]) if c_rank is not None and not pd.isna(row[c_rank]) else None
        out[nombre] = {"rank": rk, "points": pts}
    return out or None


# ============================================================
# Nivel de cada selección (para ponderar rivales y anclar)
# ============================================================
def construir_indices(df, elo, ranking=None):
    """Devuelve, por selección, un 'índice de nivel' ~1.0 = promedio.
    Si hay ranking FIFA lo usa como base (con ELO de respaldo); si no, ELO.
    Solo se consideran selecciones activas (con partidos recientes)."""
    activos = pd.unique(pd.concat([df["home_team"], df["away_team"]]))
    elo_act = np.array([elo.get(t, 1500.0) for t in activos])
    elo_med = float(np.median(elo_act))
    indices = {}
    if ranking:
        pts = np.array([ranking[t]["points"] for t in ranking])
        pts_med = float(np.median(pts)) if len(pts) else 1.0
    for t in activos:
        idx_elo = np.clip(elo.get(t, 1500.0) / elo_med, 0.55, 1.6)
        if ranking and t in ranking and ranking[t]["points"] > 0:
            idx_rank = np.clip(ranking[t]["points"] / pts_med, 0.55, 1.6)
            indices[t] = 0.6 * idx_rank + 0.4 * idx_elo
        else:
            indices[t] = idx_elo
    return indices, ("FIFA" if ranking else "ELO")


# ============================================================
# Forma reciente de una selección
# ============================================================
def _partidos_recientes(df, team, fecha_ref=None, n=N_PARTIDOS, solo_oficiales=True):
    """Últimos n partidos del equipo (perspectiva del equipo: gf, gc, rival…)."""
    m = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()
    if solo_oficiales:
        m = m[m["tournament"].astype(str) != "Friendly"]
    if fecha_ref is not None:
        m = m[m["date"] < fecha_ref]
    m = m.sort_values("date").tail(n)
    es_local = m["home_team"] == team
    return pd.DataFrame({
        "date": m["date"].values,
        "rival": np.where(es_local, m["away_team"], m["home_team"]),
        "gf": np.where(es_local, m["home_score"], m["away_score"]).astype(float),
        "gc": np.where(es_local, m["away_score"], m["home_score"]).astype(float),
        "local": es_local.astype(int).values,
        "torneo": m["tournament"].values,
    })


def expectativa_equipo(df, indices, team, fecha_ref=None, n=N_PARTIDOS):
    """Ataque y defensa esperados de `team` frente a un rival promedio,
    a partir de su forma reciente ponderada por recencia y nivel de rival.

    - Ataque: goles marcados, dando MÁS crédito a goles contra defensas
      fuertes (gf * nivel_rival).
    - Defensa: goles recibidos, DISCULPANDO los encajados contra ataques
      fuertes (gc / nivel_rival).
    """
    p = _partidos_recientes(df, team, fecha_ref, n)
    if len(p) == 0:
        return {"ataque": MU_LIGA, "defensa": MU_LIGA, "n": 0, "detalle": p}
    dias = (pd.Timestamp(fecha_ref or df["date"].max()) - pd.to_datetime(p["date"])).dt.days
    dias = np.clip(dias.values, 0, None)
    w_rec = 0.5 ** (dias / HALF_LIFE_DIAS)
    niv_rival = np.array([indices.get(r, 1.0) for r in p["rival"]])
    w = w_rec  # peso base por recencia
    sw = w.sum()
    ataque = float(np.sum(w * p["gf"].values * niv_rival) / sw)
    defensa = float(np.sum(w * p["gc"].values / niv_rival) / sw)
    detalle = p.assign(peso_recencia=np.round(w_rec, 3),
                       nivel_rival=np.round(niv_rival, 3))
    return {"ataque": np.clip(ataque, 0.15, 4.5),
            "defensa": np.clip(defensa, 0.15, 4.5),
            "n": int(len(p)), "detalle": detalle}


# ============================================================
# Historial directo (H2H)
# ============================================================
def _h2h_ajuste(df, a, b, fecha_ref=None, n=8):
    """Diferencia media de goles (A - B) en sus últimos enfrentamientos.
    Positivo => A ha dominado. Se usa como nudge pequeño sobre λ."""
    m = df[(((df["home_team"] == a) & (df["away_team"] == b)) |
            ((df["home_team"] == b) & (df["away_team"] == a)))].copy()
    if fecha_ref is not None:
        m = m[m["date"] < fecha_ref]
    m = m.sort_values("date").tail(n)
    if len(m) == 0:
        return 0.0, 0
    ga = np.where(m["home_team"] == a, m["home_score"], m["away_score"]).astype(float)
    gb = np.where(m["home_team"] == a, m["away_score"], m["home_score"]).astype(float)
    return float(np.mean(ga - gb)), int(len(m))


# ============================================================
# Matriz de marcadores con corrección de resultados bajos
# ============================================================
def _matriz_dc(lh, la, rho=RHO_BAJOS, max_goals=MAX_GOLES):
    h = poisson.pmf(np.arange(max_goals + 1), lh)
    a = poisson.pmf(np.arange(max_goals + 1), la)
    M = np.outer(h, a)
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


# ============================================================
# Predicción de un partido por forma reciente
# ============================================================
def predecir_partido_forma(df, elo, ranking, local, visitante,
                           neutral=False, n=N_PARTIDOS):
    """Pipeline completo A vs B. Devuelve probabilidades 1X2, goles esperados,
    marcadores más probables y el desglose para mostrar en la app."""
    indices, base_nivel = construir_indices(df, elo, ranking)
    fecha_ref = df["date"].max() + pd.Timedelta(days=1)

    eL = expectativa_equipo(df, indices, local, fecha_ref, n)
    eV = expectativa_equipo(df, indices, visitante, fecha_ref, n)

    # índices ataque/defensa relativos al promedio de la liga
    atk_L, def_L = eL["ataque"] / MU_LIGA, eL["defensa"] / MU_LIGA
    atk_V, def_V = eV["ataque"] / MU_LIGA, eV["defensa"] / MU_LIGA

    # --- ancla al ranking: encoge la forma hacia el nivel del ranking/ELO ---
    niv_L = indices.get(local, 1.0)
    niv_V = indices.get(visitante, 1.0)
    atk_L = (1 - BETA_ANCLA) * atk_L + BETA_ANCLA * niv_L
    def_L = (1 - BETA_ANCLA) * def_L + BETA_ANCLA * (1 / niv_L)
    atk_V = (1 - BETA_ANCLA) * atk_V + BETA_ANCLA * niv_V
    def_V = (1 - BETA_ANCLA) * def_V + BETA_ANCLA * (1 / niv_V)

    # goles esperados: ataque propio x defensa rival x media de liga
    lh = MU_LIGA * atk_L * def_V
    la = MU_LIGA * atk_V * def_L

    # ventaja de local
    if not neutral:
        lh *= GAMMA_LOCAL

    # --- nudge por historial directo ---
    h2h_diff, h2h_n = _h2h_ajuste(df, local, visitante, fecha_ref)
    lh += PESO_H2H * max(h2h_diff, 0) - PESO_H2H * max(-h2h_diff, 0) * 0.5
    la += PESO_H2H * max(-h2h_diff, 0) - PESO_H2H * max(h2h_diff, 0) * 0.5

    lh = float(np.clip(lh, 0.1, 5.5))
    la = float(np.clip(la, 0.1, 5.5))

    # matriz de marcadores con realce de resultados bajos
    M = _matriz_dc(lh, la)
    p_local = float(np.tril(M, -1).sum())
    p_empate = float(np.trace(M))
    p_visit = float(np.triu(M, 1).sum())

    # marcadores más probables
    idx = np.dstack(np.unravel_index(np.argsort(M.ravel())[::-1], M.shape))[0][:6]
    marcadores = [{"Marcador": f"{int(i)}–{int(j)}", "Prob %": round(float(M[i, j]) * 100, 1)}
                  for i, j in idx]

    return {
        "base_nivel": base_nivel,
        "prob": {"local": p_local, "empate": p_empate, "visitante": p_visit},
        "goles_esperados": {"local": round(lh, 2), "visitante": round(la, 2)},
        "marcadores": pd.DataFrame(marcadores),
        "h2h": {"diff": round(h2h_diff, 2), "n": h2h_n},
        "forma_local": eL, "forma_visitante": eV,
        "indices": {"local": round(niv_L, 3), "visitante": round(niv_V, 3)},
    }
