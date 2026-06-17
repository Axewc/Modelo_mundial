"""
scrape_fifa_ranking.py — Descarga el ranking FIFA masculino a un CSV
====================================================================
Genera `fifa_ranking.csv` (columnas: rank, team, points) que la app usa
como ancla en la pestaña de "Análisis de forma".

CÓMO USARLO (una sola vez, o cuando FIFA publique ranking nuevo):

    pip install requests pandas lxml
    python scrape_fifa_ranking.py

Si el endpoint principal de FIFA cambia, el script cae automáticamente a
Wikipedia (top de la tabla). El CSV resultante lo lee analisis_forma.py,
que ya normaliza los nombres de selección a los del CSV de resultados.

NOTA: este repo se ejecuta en un entorno sin salida de red, por eso el
scraping se hace en TU máquina y el CSV se sube al repo. La app, en
Streamlit Cloud, solo lee el CSV (no scrapea en vivo).
"""
import sys
import pandas as pd

SALIDA = "fifa_ranking.csv"

# Endpoint JSON usado por inside.fifa.com para la tabla de ranking.
FIFA_API = ("https://inside.fifa.com/api/ranking-overview"
            "?locale=en&dateId=ranking_{date_id}")
# Página de respaldo (suele traer el top de la clasificación).
WIKI_URL = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"


def _via_fifa_api():
    """Ranking completo desde la API de inside.fifa.com."""
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    # 1) descubrir el último dateId disponible
    meta = s.get("https://inside.fifa.com/api/ranking-overview?locale=en", timeout=20)
    meta.raise_for_status()
    j = meta.json()
    fechas = j.get("ranks") or j.get("dates") or []
    date_id = None
    if isinstance(j.get("rankings"), list) and j["rankings"]:
        # algunos esquemas ya devuelven el ranking directo
        return _parse_fifa_json(j)
    if fechas:
        date_id = (fechas[0].get("id") or fechas[0].get("dateId")
                   if isinstance(fechas[0], dict) else fechas[0])
    if date_id:
        r = s.get(FIFA_API.format(date_id=date_id), timeout=20)
        r.raise_for_status()
        return _parse_fifa_json(r.json())
    raise RuntimeError("No se pudo determinar el dateId del ranking FIFA")


def _parse_fifa_json(j):
    filas = j.get("rankings") or j.get("data") or []
    out = []
    for it in filas:
        team = (it.get("name") or it.get("teamName")
                or (it.get("team") or {}).get("name"))
        rank = it.get("rankingItem", {}).get("rank") if "rankingItem" in it else it.get("rank")
        pts = it.get("rankingItem", {}).get("totalPoints") if "rankingItem" in it else it.get("totalPoints") or it.get("points")
        if team and pts is not None:
            out.append({"rank": rank, "team": str(team).strip(), "points": float(pts)})
    if not out:
        raise RuntimeError("JSON de FIFA sin filas reconocibles")
    return pd.DataFrame(out)


def _via_wikipedia():
    """Respaldo: lee la primera tabla con columnas Rank/Team/Points."""
    tablas = pd.read_html(WIKI_URL)
    for t in tablas:
        cols = [str(c).lower() for c in t.columns]
        if any("team" in c for c in cols) and any("point" in c for c in cols):
            t.columns = [str(c) for c in t.columns]
            cteam = next(c for c in t.columns if "team" in c.lower())
            cpts = next(c for c in t.columns if "point" in c.lower())
            crank = next((c for c in t.columns if "rank" in c.lower()), None)
            out = pd.DataFrame({
                "rank": t[crank] if crank else range(1, len(t) + 1),
                "team": t[cteam].astype(str).str.replace(r"\[.*\]", "", regex=True).str.strip(),
                "points": pd.to_numeric(t[cpts], errors="coerce"),
            }).dropna(subset=["points"])
            return out
    raise RuntimeError("No se encontró una tabla de ranking en Wikipedia")


def main():
    df = None
    try:
        print("Intentando API de FIFA…")
        df = _via_fifa_api()
        print(f"  OK: {len(df)} selecciones desde FIFA.")
    except Exception as e:
        print(f"  Falló FIFA API ({type(e).__name__}: {e}). Probando Wikipedia…")
        try:
            df = _via_wikipedia()
            print(f"  OK: {len(df)} selecciones desde Wikipedia.")
        except Exception as e2:
            print(f"  Falló Wikipedia ({type(e2).__name__}: {e2}).", file=sys.stderr)
            print("No se pudo descargar el ranking. Revisa tu conexión o el "
                  "formato de la fuente.", file=sys.stderr)
            sys.exit(1)

    df = df.dropna(subset=["points"]).reset_index(drop=True)
    df["points"] = df["points"].astype(float)
    df.to_csv(SALIDA, index=False)
    print(f"\nGuardado {SALIDA} con {len(df)} filas. Súbelo al repo y "
          f"la app lo usará como ancla automáticamente.")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
