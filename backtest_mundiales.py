"""
backtest_mundiales.py — Backtest point-in-time en Mundiales pasados
===================================================================
La pregunta que responde: ¿cómo le habría ido a ESTE modelo (misma receta,
sin mirar el futuro) prediciendo los Mundiales 2018 y 2022, y cómo va en el
2026 en curso?

Protocolo (idéntico al de producción, desplazado en el tiempo):
  - Corte temporal en el PRIMER partido de cada Mundial: el modelo se entrena
    solo con partidos ANTERIORES a esa fecha.
  - ELO sobre toda la historia hasta el corte; Dixon-Coles sobre una ventana
    de 8 años antes del corte (la misma ventana que usa producción para 2026),
    con la regularización (alpha) elegida por validación temporal interna.
  - Ratings CONGELADOS al inicio del torneo (no se actualizan partido a
    partido durante el Mundial): es la foto que el modelo tendría el día de
    la inauguración.
  - Se evalúan TODOS los partidos jugados de ese Mundial con RPS / LogLoss /
    Brier / Accuracy, contra dos baselines sin información (uniforme y
    frecuencias históricas) y contra el ELO puro.

Nota sobre empates: siguiendo la convención del dataset, el marcador incluye
prórroga pero NO penales; un "empate" en eliminatorias significa que el
partido se decidió en tanda.

Genera:
  - docs/backtest_mundiales.md  (tablas de resultados)
  - docs/img/calibracion.png    (diagrama de fiabilidad del modelo de goles,
    walk-forward 2019-hoy; requiere matplotlib)

Uso:  python backtest_mundiales.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mundial import modelo_core as mc
from mundial.backtest import rps_set, logloss_set, brier_set

VENTANA_ANIOS = 8          # misma ventana que producción (2018 -> 2026)
CLASES = ["V", "E", "L"]   # [away, draw, home] = 0,1,2


def _outcomes(df):
    return np.where(df["home_score"] > df["away_score"], 2,
                    np.where(df["home_score"] < df["away_score"], 0, 1))


def _accuracy(P, y):
    return float(np.mean(np.argmax(P, axis=1) == np.asarray(y)))


def _fila(nombre, P, y):
    return {"Modelo": nombre, "RPS": round(rps_set(P, y), 4),
            "LogLoss": round(logloss_set(P, y), 4),
            "Brier": round(brier_set(P, y), 4),
            "Accuracy": round(_accuracy(P, y), 3)}


# ============================================================
# Backtest de un Mundial
# ============================================================
def evaluar_mundial(df_full, year, verbose=True):
    wc = df_full[(df_full["tournament"] == "FIFA World Cup")
                 & (df_full["date"].dt.year == year)].copy()
    if wc.empty:
        return None
    corte = wc["date"].min()
    hist = df_full[df_full["date"] < corte]
    if verbose:
        print(f"\n=== Mundial {year} · corte {corte.date()} · "
              f"{len(wc)} partidos · entreno con {len(hist)} previos ===")

    # --- entrenamiento point-in-time (misma receta que producción) ---
    df_win, elo = mc.compute_elo(hist, anio_inicio=year - VENTANA_ANIOS)
    try:
        a_cut = int(len(df_win) * 0.85)
        alpha = mc.select_alpha_dc(df_win.iloc[:a_cut], df_win.iloc[a_cut:])
    except Exception:
        alpha = 0.05
    dc = mc.fit_dc(df_win, alpha=alpha)

    fb = ["elo_diff", "home_advantage", "neutral_match"]
    blr = LogisticRegression(max_iter=1000).fit(df_win[fb], _outcomes(df_win))

    # --- predicciones (ratings congelados al inicio del torneo) ---
    y = _outcomes(wc)
    neutral = wc["neutral"].astype(str).str.lower().isin(["true", "1"]).values

    P_dc = np.array([mc.dc_result_probs(dc, h, a, bool(n)) for h, a, n in
                     zip(wc["home_team"], wc["away_team"], neutral)])
    X = pd.DataFrame({
        "elo_diff": [elo.get(h, 1500.0) - elo.get(a, 1500.0)
                     for h, a in zip(wc["home_team"], wc["away_team"])],
        "home_advantage": (1 - neutral.astype(int)),
        "neutral_match": neutral.astype(int)})
    P_elo = blr.predict_proba(X)[:, np.argsort(blr.classes_)]
    P_ens = (P_dc + P_elo) / 2.0

    # --- baselines sin información ---
    P_uni = np.full((len(wc), 3), 1 / 3)
    frec = np.bincount(_outcomes(df_win), minlength=3) / len(df_win)
    P_frec = np.tile(frec, (len(wc), 1))

    tabla = pd.DataFrame([
        _fila("Ensemble (DC+ELO)", P_ens, y),
        _fila("Dixon-Coles", P_dc, y),
        _fila("ELO logístico", P_elo, y),
        _fila("Baseline frecuencias", P_frec, y),
        _fila("Baseline uniforme", P_uni, y),
    ]).sort_values("RPS").reset_index(drop=True)
    if verbose:
        print(tabla.to_string(index=False))
    return {"year": year, "n": len(wc), "corte": str(corte.date()),
            "tabla": tabla, "P_ens": P_ens, "P_dc": P_dc, "y": y}


# ============================================================
# Walk-forward global (para el diagrama de fiabilidad)
# ============================================================
def walk_forward_dc(df_full, desde=2019, retrain_every=300, verbose=True):
    """Predicciones point-in-time del modelo de goles (Dixon-Coles) sobre
    todos los partidos desde `desde`, reentrenando cada `retrain_every`
    partidos con una ventana móvil de VENTANA_ANIOS años. En cada refit la
    regularización (alpha) se elige por validación interna, igual que en
    producción."""
    d = df_full.copy()
    d["neutral_match"] = (d["neutral"].astype(str).str.lower()
                          .isin(["true", "1"]).astype(int))
    idx_eval = d.index[d["date"] >= f"{desde}-01-01"].tolist()
    P, ys = [], []
    model, last_fit = None, -10 ** 9
    for pos in idx_eval:
        if model is None or (pos - last_fit) >= retrain_every:
            fecha = d.loc[pos, "date"]
            train = d[(d["date"] < fecha)
                      & (d["date"] >= fecha - pd.DateOffset(years=VENTANA_ANIOS))]
            if len(train) < 500:
                continue
            cut = int(len(train) * 0.85)
            try:
                alpha = mc.select_alpha_dc(train.iloc[:cut], train.iloc[cut:])
            except Exception:
                alpha = 0.01
            model = mc.fit_dc(train, alpha=alpha)
            last_fit = pos
        row = d.loc[pos]
        try:
            p = mc.dc_result_probs(model, row["home_team"], row["away_team"],
                                   bool(row["neutral_match"]))
        except Exception:
            continue
        P.append(p)
        ys.append(2 if row["home_score"] > row["away_score"]
                  else (0 if row["home_score"] < row["away_score"] else 1))
    if verbose:
        print(f"\nWalk-forward {desde}-hoy: {len(ys)} partidos point-in-time")
    return np.array(P), np.array(ys, int)


# ============================================================
# Diagrama de fiabilidad (calibración)
# ============================================================
def fig_calibracion(P, y, path="docs/img/calibracion.png", bins=10, min_n=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2 = "#fcfcfb", "#333333", "#666666"
    # offsets de etiqueta por serie para evitar colisiones en el extremo
    series = [("Gana local", 2, "#2a78d6", (8, -11)),
              ("Empate", 1, "#1baf7a", (8, 2)),
              ("Gana visitante", 0, "#eda100", (-4, 12))]

    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#bbbbbb", zorder=1)

    edges = np.linspace(0, 1, bins + 1)
    for nombre, cls, color, off in series:
        p = P[:, cls]
        obs = (y == cls).astype(float)
        xs, os_ = [], []
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1])
            if m.sum() >= min_n:
                xs.append(p[m].mean()); os_.append(obs[m].mean())
        ax.plot(xs, os_, "-o", lw=2, ms=6, color=color, zorder=3)
        if xs:  # etiqueta directa al final de la línea (regla de relief)
            ax.annotate(nombre, (xs[-1], os_[-1]), xytext=off,
                        textcoords="offset points", va="center",
                        fontsize=9, color=INK)

    ax.set_xlim(0, 1.18); ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Probabilidad predicha", color=INK2, fontsize=10)
    ax.set_ylabel("Frecuencia observada", color=INK2, fontsize=10)
    ax.set_title("Calibración del modelo de goles (Dixon-Coles)\n"
                 f"predicciones point-in-time 2019–hoy · n={len(y):,} partidos",
                 loc="left", fontsize=11, color=INK)
    ax.grid(True, lw=0.5, color="#e8e8e6", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=INK2, labelsize=9)
    fig.text(0.01, 0.01, "La diagonal punteada = calibración perfecta. "
             f"Bins de {1/bins:.0%} con n≥{min_n}.", fontsize=8, color=INK2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# Main: corre todo y escribe docs/
# ============================================================
PREDS_CACHE = "docs/walkforward_preds.csv"


def main(solo_fig=False):
    if solo_fig and os.path.exists(PREDS_CACHE):
        c = pd.read_csv(PREDS_CACHE)
        P_wf = c[["p_away", "p_draw", "p_home"]].to_numpy()
        y_wf = c["y"].to_numpy(int)
        print(f"Redibujando desde {PREDS_CACHE} ({len(y_wf)} predicciones)")
        print("→", fig_calibracion(P_wf, y_wf))
        return

    df_full, _ = mc.load_data()
    resultados = []
    for year in (2018, 2022, 2026):
        r = evaluar_mundial(df_full, year)
        if r:
            resultados.append(r)

    P_wf, y_wf = walk_forward_dc(df_full)
    os.makedirs("docs", exist_ok=True)
    pd.DataFrame({"p_away": P_wf[:, 0], "p_draw": P_wf[:, 1],
                  "p_home": P_wf[:, 2], "y": y_wf}).to_csv(PREDS_CACHE,
                                                           index=False)
    fig = fig_calibracion(P_wf, y_wf)
    tabla_wf = pd.DataFrame([_fila("Dixon-Coles (walk-forward)", P_wf, y_wf),
                             _fila("Baseline uniforme",
                                   np.full((len(y_wf), 3), 1 / 3), y_wf)])

    os.makedirs("docs", exist_ok=True)
    with open("docs/backtest_mundiales.md", "w") as f:
        f.write("# Backtest point-in-time — Mundiales\n\n")
        f.write("Generado por `python backtest_mundiales.py`. Protocolo: "
                "entrenamiento solo con partidos anteriores al torneo, "
                "ratings congelados al inicio, misma receta que producción "
                f"(ventana {VENTANA_ANIOS} años). El marcador incluye prórroga "
                "pero no penales.\n")
        for r in resultados:
            estado = " (en curso)" if r["year"] == 2026 else ""
            f.write(f"\n## Mundial {r['year']}{estado} — {r['n']} partidos "
                    f"(corte {r['corte']})\n\n")
            f.write(r["tabla"].to_markdown(index=False) + "\n")
        f.write(f"\n## Walk-forward global 2019–hoy — {len(y_wf):,} partidos\n\n")
        f.write(tabla_wf.to_markdown(index=False) + "\n")
        f.write(f"\n![Calibración]({os.path.relpath(fig, 'docs')})\n")
    print(f"\nEscrito docs/backtest_mundiales.md y {fig}")


if __name__ == "__main__":
    import sys
    main(solo_fig="--solo-fig" in sys.argv)
