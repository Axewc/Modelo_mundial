"""
update_results.py — Sincroniza el dataset de resultados con el upstream
=======================================================================
Refresca `results_actualizado_mundial_2026.csv` con la versión más reciente
del dataset comunitario `martj42/international_results` (1872-presente, mismo
esquema), CONSERVANDO los fixtures futuros del Mundial 2026 que tú ya tienes
(las filas sin marcador). Así no pierdes los partidos por jugar y rellenas
todos los resultados nuevos de una sola vez.

CÓMO USARLO (en tu máquina, hay que tener red):

    pip install requests pandas
    python update_results.py            # update normal (conserva tus filas)
    python update_results.py --limpio   # update LIMPIO (ver abajo)
    python update_results.py --dry-run  # solo muestra qué cambiaría
    python update_results.py --limpio --dry-run   # combínalos

MODO NORMAL vs. MODO LIMPIO
---------------------------
- Normal: el upstream corrige los marcadores de los partidos que ambos tienen,
  y se conservan TODAS tus filas extra (fixtures futuros + cualquier resultado
  que metiste a mano). Útil para el día a día.
- Limpio (`--limpio`): el upstream es la ÚNICA fuente de verdad para partidos
  jugados. De tu CSV se conservan SOLO los fixtures futuros (sin marcador) y se
  DESCARTAN los resultados tuyos que el upstream no tenga. Úsalo cuando
  sospeches que la ingesta manual se corrompió y quieras reconstruir limpio.

En cualquier modo se imprime un reporte de los marcadores que DIFIEREN entre tu
CSV y el upstream (la señal de corrupción real).

Luego haz commit del CSV y súbelo al repo (igual que con el ranking FIFA).

NOTA: este entorno no tiene salida de red; por eso la descarga se hace en TU
máquina. La función `fusionar` es pura y testeable sin red.
"""
import sys
import argparse
import pandas as pd

ARCHIVO = "results_actualizado_mundial_2026.csv"
# Dataset canónico de resultados de selecciones (mismo esquema de columnas).
UPSTREAM_URL = ("https://raw.githubusercontent.com/martj42/"
                "international_results/master/results.csv")
COLS = ["date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral"]
CLAVE = ["date", "home_team", "away_team"]


def _norm(df):
    """Asegura columnas/orden y fecha como YYYY-MM-DD (texto estable)."""
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = pd.NA
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return d[COLS]


def _diferencias(a, u):
    """Marcadores que DIFIEREN entre tu CSV y el upstream en partidos que ambos
    tienen (misma clave) y que ambos reportan como jugados. Es la señal de que
    la ingesta manual quedó corrompida."""
    am = a.merge(u, on=CLAVE, suffixes=("_local", "_up"))
    hl = pd.to_numeric(am["home_score_local"], errors="coerce")
    al = pd.to_numeric(am["away_score_local"], errors="coerce")
    hu = pd.to_numeric(am["home_score_up"], errors="coerce")
    au = pd.to_numeric(am["away_score_up"], errors="coerce")
    ambos = hl.notna() & hu.notna()
    dif = ambos & ((hl != hu) | (al != au))
    return am.loc[dif, CLAVE + ["home_score_local", "away_score_local",
                                "home_score_up", "away_score_up"]]


def fusionar(actual, upstream, limpio=False):
    """Une el CSV actual con el upstream.

    Modo normal (limpio=False):
      - El upstream manda en los partidos JUGADOS que ambos tienen.
      - Se CONSERVAN todas las filas del actual que el upstream no tiene
        (fixtures futuros Y resultados que metiste a mano).

    Modo limpio (limpio=True) -> reconstrucción confiable:
      - El upstream es la ÚNICA fuente de verdad para partidos jugados.
      - Del actual solo se conservan los FIXTURES futuros (sin marcador) que el
        upstream todavía no publica. Cualquier resultado tuyo ausente en el
        upstream se DESCARTA (evita arrastrar marcadores corrompidos).

    Devuelve (resultado, stats, diffs, descartados)."""
    a, u = _norm(actual), _norm(upstream)
    a_keys = a.set_index(CLAVE).index
    u_keys = u.set_index(CLAVE).index

    solo_actual = a[~a_keys.isin(u_keys)]
    es_fixture = solo_actual["home_score"].isna()
    descartados = solo_actual[~es_fixture]            # resultados tuyos no en upstream
    if limpio:
        solo_actual = solo_actual[es_fixture]         # conserva solo fixtures futuros

    fixtures = int(solo_actual["home_score"].isna().sum())
    diffs = _diferencias(a, u)

    out = (pd.concat([u, solo_actual], ignore_index=True)
           .drop_duplicates(subset=CLAVE, keep="first")
           .sort_values(["date", "home_team", "away_team"])
           .reset_index(drop=True))
    stats = {
        "actual": len(a), "upstream": len(u), "resultado": len(out),
        "conservados_del_actual": len(solo_actual),
        "fixtures_conservados": fixtures,
        "descartados_con_marcador": len(descartados) if limpio else 0,
        "marcadores_distintos": len(diffs),
        "partidos_agregados": len(out) - len(a),
    }
    return out, stats, diffs, descartados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribe; solo muestra el resumen.")
    ap.add_argument("--limpio", action="store_true",
                    help="Reconstrucción limpia: descarta tus resultados que el "
                         "upstream no tenga (conserva solo fixtures futuros).")
    ap.add_argument("--url", default=UPSTREAM_URL)
    ap.add_argument("--archivo", default=ARCHIVO)
    args = ap.parse_args()

    import requests
    from io import StringIO
    try:
        print("Descargando upstream…")
        r = requests.get(args.url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        upstream = pd.read_csv(StringIO(r.text))
        print(f"  upstream: {len(upstream)} partidos.")
    except Exception as e:
        print(f"  Falló la descarga ({type(e).__name__}: {e})", file=sys.stderr)
        sys.exit(1)

    actual = pd.read_csv(args.archivo)
    out, s, diffs, descartados = fusionar(actual, upstream, limpio=args.limpio)

    modo = "LIMPIO" if args.limpio else "normal"
    print(f"\nModo: {modo}")
    print(f"Actual: {s['actual']}  |  Upstream: {s['upstream']}  |  "
          f"Resultado: {s['resultado']}")
    print(f"  + {s['partidos_agregados']} partidos nuevos respecto a tu CSV")
    print(f"  conservados solo-tuyos: {s['conservados_del_actual']} "
          f"(de ellos {s['fixtures_conservados']} fixtures futuros)")

    if s["marcadores_distintos"]:
        print(f"\n⚠ {s['marcadores_distintos']} partido(s) con marcador DISTINTO "
              f"al upstream (posible corrupción). El upstream prevalece:")
        print(diffs.head(40).to_string(index=False))
        if len(diffs) > 40:
            print(f"  … y {len(diffs) - 40} más.")

    if args.limpio and s["descartados_con_marcador"]:
        print(f"\n🗑 {s['descartados_con_marcador']} resultado(s) tuyos NO presentes "
              f"en el upstream se DESCARTAN en modo limpio:")
        print(descartados[CLAVE + ["home_score", "away_score", "tournament"]]
              .head(40).to_string(index=False))
        if len(descartados) > 40:
            print(f"  … y {len(descartados) - 40} más.")
    elif not args.limpio and len(descartados):
        print(f"\nℹ {len(descartados)} resultado(s) tuyos no están en el upstream "
              f"y se CONSERVAN (usa --limpio para descartarlos).")

    if args.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return
    out.to_csv(args.archivo, index=False)
    print(f"\nGuardado {args.archivo}. Haz commit y súbelo al repo.")


if __name__ == "__main__":
    main()
