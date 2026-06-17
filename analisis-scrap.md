# Análisis de scraping — ranking FIFA masculino

## Estado actual (junio 2026)

El script `scrape_fifa_ranking.py` funciona correctamente usando la API interna de FIFA descubierta inspeccionando el JS del sitio.

---

## APIs que funcionan

### 1. `inside.fifa.com` — descubrimiento del `dateId`

**Propósito:** Obtener el ID de la fecha más reciente del ranking.

**URL:**

```http
GET https://inside.fifa.com/fifa-world-ranking/men
```

**Tipo de respuesta:** HTML (página Next.js renderizada en servidor)

**Cómo extraer el dateId:**

El HTML incluye un JSON embebido con la clave `allAvailableDates`. Se puede extraer con regex:

```python
import re, json, requests

r = requests.get("https://inside.fifa.com/fifa-world-ranking/men",
                  headers={"User-Agent": "Mozilla/5.0"})
m = re.search(r'"allAvailableDates"\s*:\s*(\[[^\]]*\])', r.text)
dates = json.loads(m.group(1))
date_id = dates[0]["id"]  # el primero siempre es el más reciente
# → "FRS_Male_Football_20260401"
```

**Estructura del array:**

```json
[
  {"id": "FRS_Male_Football_20260401", "date": "2026-06-11", "matchWindowEndDate": "2026-06-11"},
  {"id": "FRS_Male_Football_20260119", "date": "2026-04-01", "matchWindowEndDate": "2026-04-01"},
  ...
]
```

**Formato del `id`:** `FRS_Male_Football_YYYYMMDD` donde la fecha es la de publicación del ranking.

---

### 2. `api.fifa.com/api/v3` — tabla de rankings (FDCP API)

**Propósito:** Descargar la tabla completa de rankings con puntos para un `dateId` dado.

**URL:**

```http
GET https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule
    ?rankingScheduleId=FRS_Male_Football_20260401
    &language=en
```

**Headers requeridos:**

```http
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36
Accept: application/json
Origin: https://inside.fifa.com
Referer: https://inside.fifa.com/
```

**No requiere auth** (la URL pública sin `forceProdApi=1` usa la base URL hardcoded en el JS).

**Ejemplo de respuesta:**

```json
{
  "ContinuationToken": null,
  "ContinuationHash": null,
  "Results": [
    {
      "IdTeam": "43922",
      "TeamName": [{"Locale": "en-GB", "Description": "Argentina"}],
      "Gender": 1,
      "IdCountry": "ARG",
      "Rank": 1,
      "PrevRank": 3,
      "TotalPoints": 1877.266571,
      "PrevPoints": 1874.814835,
      "RatedMatches": 60,
      "RankingMovement": 2,
      "ConfederationName": "CONMEBOL"
    },
    ...
  ]
}
```

**Campos útiles por ítem:**

| Campo | Tipo | Descripción |
|---|---|---|
| `Rank` | int | Posición actual |
| `PrevRank` | int | Posición anterior |
| `TotalPoints` | float | Puntos FIFA actuales |
| `PrevPoints` | float | Puntos del ranking anterior |
| `TeamName[].Description` | str | Nombre del equipo (filtrar por `Locale` con `"en"`) |
| `IdCountry` | str | Código ISO del país (ej. `"ARG"`) |
| `ConfederationName` | str | Confederación (`CONMEBOL`, `UEFA`, etc.) |

**Ejemplo completo de consulta:**

```python
import requests, pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Origin": "https://inside.fifa.com",
    "Referer": "https://inside.fifa.com/",
}

date_id = "FRS_Male_Football_20260401"
url = (f"https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
       f"?rankingScheduleId={date_id}&language=en")

r = requests.get(url, headers=HEADERS, timeout=30)
results = r.json()["Results"]

rows = []
for it in results:
    names = it.get("TeamName", [])
    name = next((n["Description"] for n in names if "en" in n.get("Locale","").lower()),
                names[0]["Description"] if names else None)
    rows.append({"rank": it["Rank"], "team": name, "points": it["TotalPoints"]})

df = pd.DataFrame(rows).sort_values("rank")
print(df.head(10))
```

---

## APIs que NO funcionan (deprecadas)

| Endpoint | Problema |
|---|---|
| `inside.fifa.com/api/ranking-overview?locale=en` | Devuelve `{"rankings":[]}` — endpoint muerto |
| `inside.fifa.com/api/ranking-overview?locale=en&dateId=ranking_20240620` | Formato de dateId obsoleto; igual devuelve lista vacía |
| `api.fifa.com/api/v3/ranking/men` | HTTP 404 |
| `pd.read_html("https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking")` | La página no tiene tabla con columna "points"; solo tablas históricas |

---

## Cómo funciona internamente

FIFA migró su arquitectura a un sistema llamado **FDCP** (*FIFA Data Content Platform*). El JS de `inside.fifa.com` expone dos entornos:

- **Dev/público:** `https://api.fifa.com/api/v3` (hardcoded en el bundle JS — **usamos este**)
- **Prod:** URL guardada en la variable de entorno privada `NEXT_PUBLIC_FIFA_FDCP_API_PROD` (no expuesta, requeriría bypass)

El client ID del sitio es `64e9afa8-c5c0-413d-882b-bc9e6a81e264` pero no es necesario para las consultas de ranking público.

---

## Notas de mantenimiento

- El `dateId` cambia cada vez que FIFA publica un nuevo ranking (aprox. cada 2 meses). El script descubre el más reciente automáticamente.
- Si el script falla en el futuro, verificar primero que `allAvailableDates` siga presente en el HTML de `inside.fifa.com/fifa-world-ranking/men`.
- Si FIFA cambia el bundle JS, la URL base (`api.fifa.com/api/v3`) podría cambiar. Se puede buscar en `/_next/static/chunks/6569-*.js` la cadena `NEXT_PUBLIC_FIFA_FDCP_API`.
