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
    python update_results.py            # actualiza el CSV in-place
    python update_results.py --dry-run  # solo muestra qué cambiaría

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


def fusionar(actual, upstream):
    """Une el CSV actual con el upstream. Reglas:
      - El upstream manda en los partidos JUGADOS (corrige marcadores).
      - Se CONSERVAN las filas del actual que el upstream no tiene (p.ej. los
        fixtures futuros del Mundial, o partidos que metiste a mano y que el
        upstream aún no publica).
    Devuelve (resultado, stats)."""
    a, u = _norm(actual), _norm(upstream)
    a_keys = a.set_index(CLAVE).index
    u_keys = u.set_index(CLAVE).index
    solo_actual = a[~a_keys.isin(u_keys)]
    fixtures = int(solo_actual["home_score"].isna().sum())
    out = (pd.concat([u, solo_actual], ignore_index=True)
           .drop_duplicates(subset=CLAVE, keep="first")
           .sort_values(["date", "home_team", "away_team"])
           .reset_index(drop=True))
    stats = {
        "actual": len(a), "upstream": len(u), "resultado": len(out),
        "conservados_del_actual": len(solo_actual),
        "fixtures_conservados": fixtures,
        "partidos_agregados": len(out) - len(a),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribe; solo muestra el resumen.")
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
    out, s = fusionar(actual, upstream)
    print(f"\nActual: {s['actual']}  |  Upstream: {s['upstream']}  |  "
          f"Resultado: {s['resultado']}")
    print(f"  + {s['partidos_agregados']} partidos nuevos respecto a tu CSV")
    print(f"  conservados solo-tuyos: {s['conservados_del_actual']} "
          f"(de ellos {s['fixtures_conservados']} fixtures futuros)")

    if args.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return
    out.to_csv(args.archivo, index=False)
    print(f"\nGuardado {args.archivo}. Haz commit y súbelo al repo.")


if __name__ == "__main__":
    main()
