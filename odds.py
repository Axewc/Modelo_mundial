"""
odds.py — Cuotas de casas de apuestas: comparativo y blend con el modelo
========================================================================
Trae las cuotas 1X2 de The Odds API (key en st.secrets [odds]), las convierte
a probabilidad implícita (quitando el margen de la casa) y permite:

  - COMPARAR la probabilidad del mercado vs la del modelo (y marcar "valor",
    donde el modelo da más probabilidad que el mercado).
  - MEZCLAR (blend) ambas por pool logarítmico, con un peso al mercado.

El mercado es el mejor benchmark público; el blend suele mejorar calibración
a costa de "seguir" al mercado. Sin key configurada, la app deja ingresar las
cuotas a mano (estas funciones no dependen de red salvo `traer_cuotas`).

Convención de orden: [away, draw, home] = CLASSES [0,1,2] del modelo.
"""
import numpy as np
import pandas as pd

try:
    from ingesta import normaliza_equipo
except Exception:  # módulo aislado / tests
    def normaliza_equipo(n):
        return str(n).strip()

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_DEFAULT = "soccer_fifa_world_cup"


# ============================================================
# Probabilidad implícita (quita el margen / overround)
# ============================================================
def implied_probs(home_odds, draw_odds, away_odds):
    """Cuotas decimales 1/X/2 -> prob implícita normalizada (sin margen).
    Devuelve [away, draw, home]."""
    odds = np.array([away_odds, draw_odds, home_odds], dtype=float)
    if np.any(odds <= 1.0) or np.any(~np.isfinite(odds)):
        raise ValueError("Cuotas decimales inválidas (deben ser > 1).")
    inv = 1.0 / odds
    return inv / inv.sum()


def overround(home_odds, draw_odds, away_odds):
    """Margen de la casa: suma de inversas - 1 (p.ej. 0.05 = 5%)."""
    return float(1 / home_odds + 1 / draw_odds + 1 / away_odds - 1)


# ============================================================
# Blend modelo + mercado (pool logarítmico)
# ============================================================
def blend(model_p, market_p, w_market=0.5):
    """Mezcla por pool logarítmico: p ∝ model^(1-w) * market^w, normalizado.
    w_market=0 -> solo modelo; 1 -> solo mercado. Entrada/salida [away,draw,home]."""
    m = np.clip(np.asarray(model_p, float), 1e-12, 1)
    k = np.clip(np.asarray(market_p, float), 1e-12, 1)
    w = float(np.clip(w_market, 0, 1))
    logp = (1 - w) * np.log(m) + w * np.log(k)
    p = np.exp(logp - logp.max())
    return p / p.sum()


# ============================================================
# Parseo de la respuesta de The Odds API (aislado para testear sin red)
# ============================================================
def parse_eventos(eventos):
    """Convierte el JSON de /odds en filas con la implícita media de las casas.
    Promedia las cuotas decimales de todas las bookmakers por desenlace."""
    filas = []
    for ev in eventos or []:
        home = normaliza_equipo(ev.get("home_team", ""))
        away = normaliza_equipo(ev.get("away_team", ""))
        if not home or not away:
            continue
        h, d, a, nb = [], [], [], 0
        for bk in ev.get("bookmakers", []):
            mk = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
            if not mk:
                continue
            precios = {normaliza_equipo(o.get("name", "")): o.get("price")
                       for o in mk.get("outcomes", [])}
            ph, pa = precios.get(home), precios.get(away)
            pd_ = precios.get("Draw") or precios.get("Empate")
            if ph and pa and pd_:
                h.append(ph); d.append(pd_); a.append(pa); nb += 1
        if nb == 0:
            continue
        ho, do, ao = np.median(h), np.median(d), np.median(a)
        try:
            p = implied_probs(ho, do, ao)
        except ValueError:
            continue
        filas.append({
            "commence_time": ev.get("commence_time"),
            "home_team": home, "away_team": away,
            "cuota_home": round(float(ho), 2), "cuota_draw": round(float(do), 2),
            "cuota_away": round(float(ao), 2),
            "p_home": float(p[2]), "p_draw": float(p[1]), "p_away": float(p[0]),
            "margen": round(overround(ho, do, ao) * 100, 1), "n_casas": nb,
        })
    return pd.DataFrame(filas)


def traer_cuotas(api_key, sport=SPORT_DEFAULT, regions="eu,uk", session=None):
    """Descarga y parsea las cuotas 1X2 del deporte indicado. Requiere red."""
    import requests
    s = session or requests
    r = s.get(f"{API_BASE}/sports/{sport}/odds",
              params={"apiKey": api_key, "regions": regions,
                      "markets": "h2h", "oddsFormat": "decimal"},
              timeout=20)
    r.raise_for_status()
    return parse_eventos(r.json())


def buscar_partido(cuotas_df, local, visitante):
    """Busca en el DataFrame de cuotas el partido local-visitante (en cualquier
    orden) y devuelve la fila o None."""
    if cuotas_df is None or cuotas_df.empty:
        return None
    L, V = normaliza_equipo(local), normaliza_equipo(visitante)
    m = cuotas_df[((cuotas_df["home_team"] == L) & (cuotas_df["away_team"] == V)) |
                  ((cuotas_df["home_team"] == V) & (cuotas_df["away_team"] == L))]
    return None if m.empty else m.iloc[0]
