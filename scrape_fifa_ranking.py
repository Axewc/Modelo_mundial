"""
scrape_fifa_ranking.py — Descarga el ranking FIFA masculino a un CSV
====================================================================
Genera `fifa_ranking.csv` (columnas: rank, team, points) que la app usa
como ancla en la pestaña de "Análisis de forma".

CÓMO USARLO (una sola vez, o cuando FIFA publique ranking nuevo):

    pip install requests pandas
    python scrape_fifa_ranking.py

El script usa la API interna de inside.fifa.com (descubierta inspeccionando
el JS de la página). No requiere auth ni clave privada.

NOTA: este repo se ejecuta en un entorno sin salida de red, por eso el
scraping se hace en TU máquina y el CSV se sube al repo. La app, en
Streamlit Cloud, solo lee el CSV (no scrapea en vivo).
"""
import sys
import re
import json
import pandas as pd

SALIDA = "fifa_ranking.csv"

# URL del sitio de FIFA (para descubrir el último dateId)
INSIDE_FIFA_RANKING_URL = "https://inside.fifa.com/fifa-world-ranking/men"
# Base URL de la API FDCP (hardcoded en el JS de inside.fifa.com)
FDCP_BASE = "https://api.fifa.com/api/v3"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Origin": "https://inside.fifa.com",
    "Referer": "https://inside.fifa.com/",
}


def _get_latest_date_id(session):
    """Lee la página de FIFA y extrae el ID de la fecha más reciente."""
    r = session.get(INSIDE_FIFA_RANKING_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    m = re.search(r'"allAvailableDates"\s*:\s*(\[[^\]]*\])', r.text)
    if not m:
        raise RuntimeError("No se encontró allAvailableDates en la página de FIFA")
    dates = json.loads(m.group(1))
    if not dates:
        raise RuntimeError("La lista allAvailableDates está vacía")
    # El primer elemento es siempre el más reciente
    date_id = dates[0].get("id") or dates[0].get("dateId")
    if not date_id:
        raise RuntimeError(f"Estructura inesperada en allAvailableDates: {dates[0]}")
    return date_id


def _fetch_rankings(session, date_id):
    """Descarga todos los rankings para el dateId dado."""
    url = (f"{FDCP_BASE}/fifarankings/rankings/rankingsbyschedule"
           f"?rankingScheduleId={date_id}&language=en")
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    results = j.get("Results") or j.get("results") or []
    if not results:
        raise RuntimeError(f"La API devolvió 0 resultados para dateId={date_id}")
    rows = []
    for it in results:
        # TeamName es una lista [{Locale, Description}, ...]
        names = it.get("TeamName") or []
        name_en = next(
            (n["Description"] for n in names if "en" in n.get("Locale", "").lower()),
            names[0]["Description"] if names else None,
        )
        rank = it.get("Rank")
        pts = it.get("TotalPoints") or it.get("DecimalTotalPoints")
        if name_en and pts is not None:
            rows.append({"rank": rank, "team": str(name_en).strip(), "points": float(pts)})
    if not rows:
        raise RuntimeError("No se pudo parsear ninguna fila del JSON de FIFA")
    return pd.DataFrame(rows)


def main():
    import requests
    session = requests.Session()
    try:
        print("Buscando fecha más reciente del ranking FIFA…")
        date_id = _get_latest_date_id(session)
        print(f"  Usando dateId: {date_id}")
        print("Descargando rankings…")
        df = _fetch_rankings(session, date_id)
        print(f"  OK: {len(df)} selecciones.")
    except Exception as e:
        print(f"  Falló ({type(e).__name__}: {e})", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=["points"]).sort_values("rank").reset_index(drop=True)
    df["points"] = df["points"].astype(float)
    df.to_csv(SALIDA, index=False)
    print(f"\nGuardado {SALIDA} con {len(df)} filas. Súbelo al repo y "
          f"la app lo usará como ancla automáticamente.")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
