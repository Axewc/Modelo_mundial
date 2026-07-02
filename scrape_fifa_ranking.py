"""
scrape_fifa_ranking.py — Descarga el ranking FIFA masculino a un CSV
====================================================================
Genera `fifa_ranking.csv` (columnas: rank, team, points) que la app usa
como ancla en la pestaña de "Análisis de forma".

Por defecto usa el ranking LIVE (en vivo), que coincide con lo que muestra
inside.fifa.com durante/tras partidos recientes. Si quieres un corte oficial
publicado, usa `--mode official` (y opcionalmente `--date YYYY-MM-DD`).

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
import argparse
from datetime import datetime, timezone
import pandas as pd

SALIDA = "fifa_ranking.csv"

# Base URL de la API FDCP (hardcoded en el JS de inside.fifa.com)
FDCP_BASE = "https://api.fifa.com/api/v3"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Origin": "https://inside.fifa.com",
    "Referer": "https://inside.fifa.com/",
}


def _parse_iso_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _get_schedules(session):
    """Obtiene todas las fechas de ranking masculino desde el endpoint oficial."""
    url = f"{FDCP_BASE}/rankingschedules/all?type=0&gender=male"
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    schedules = j.get("Results") or j.get("results") or []
    if not schedules:
        raise RuntimeError("La API devolvió 0 schedules para ranking masculino")
    return schedules


def _pick_schedule_id(schedules, requested_date=None):
    """Selecciona el schedule_id más reciente visible o uno por fecha explícita."""
    if requested_date:
        for sch in schedules:
            official = _parse_iso_date(sch.get("OfficialDate"))
            if official and official.date().isoformat() == requested_date:
                return sch.get("IdRankingSchedule"), official.date().isoformat()
        raise RuntimeError(f"No existe ranking con fecha oficial {requested_date}")

    now = datetime.now(timezone.utc)
    candidates = []
    for sch in schedules:
        schedule_id = sch.get("IdRankingSchedule")
        official = _parse_iso_date(sch.get("OfficialDate"))
        visible = _parse_iso_date(sch.get("VisibilityDate"))
        if not schedule_id or not official:
            continue
        if official <= now and (visible is None or visible <= now):
            candidates.append((official, schedule_id))

    if not candidates:
        raise RuntimeError("No se encontró ningún ranking oficial visible")

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked_date, picked_id = candidates[0]
    return picked_id, picked_date.date().isoformat()


def _fetch_rankings(session, schedule_id):
    """Descarga todos los rankings para el schedule_id dado."""
    endpoints = [
        f"{FDCP_BASE}/rankingsbyschedule?rankingScheduleId={schedule_id}",
        f"{FDCP_BASE}/fifarankings/rankings/rankingsbyschedule?rankingScheduleId={schedule_id}&language=en",
    ]
    results = []
    for url in endpoints:
        r = session.get(url, headers=HEADERS, timeout=30)
        if r.status_code >= 400:
            continue
        j = r.json()
        results = j.get("Results") or j.get("results") or []
        if results:
            break
    if not results:
        raise RuntimeError(f"La API devolvió 0 resultados para scheduleId={schedule_id}")
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


def _fetch_live_rankings(session):
    """Descarga el ranking LIVE (en vivo) masculino."""
    url = f"{FDCP_BASE}/fifarankings/rankings/live?gender=male&sportType=0&language=en"
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    results = j.get("Results") or j.get("results") or []
    if not results:
        raise RuntimeError("La API LIVE devolvió 0 resultados")

    rows = []
    for it in results:
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
        raise RuntimeError("No se pudo parsear ninguna fila del JSON LIVE de FIFA")
    return pd.DataFrame(rows)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga ranking FIFA masculino (live u official) a fifa_ranking.csv"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "official"],
        default="live",
        help="Fuente del ranking: live (en vivo) u official (corte publicado)",
    )
    parser.add_argument(
        "--date",
        help="Solo con --mode official. Fecha oficial en formato YYYY-MM-DD (ej. 2026-04-01)",
    )
    return parser.parse_args()


def main():
    import requests
    args = _parse_args()
    session = requests.Session()
    try:
        if args.mode == "live":
            if args.date:
                raise RuntimeError("--date solo se puede usar con --mode official")
            print("Descargando ranking LIVE (en vivo) FIFA masculino…")
            df = _fetch_live_rankings(session)
        else:
            print("Buscando fecha oficial del ranking FIFA masculino…")
            schedules = _get_schedules(session)
            schedule_id, official_date = _pick_schedule_id(schedules, requested_date=args.date)
            print(f"  Usando scheduleId: {schedule_id} (fecha oficial: {official_date})")
            print("Descargando rankings oficiales…")
            df = _fetch_rankings(session, schedule_id)
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
