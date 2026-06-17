# Modelo predictivo — Mundial 2026

App de Streamlit que ejecuta el pipeline completo (ELO, Poisson, XGBoost,
Dixon-Coles) y simula el torneo con el **cuadro oficial FIFA 2026**.

## Archivos
- `app.py` — interfaz Streamlit (6 pestañas).
- `modelo_core.py` — todo el modelo (ingesta, ELO, Dixon-Coles, simulador).
- `analisis_forma.py` — análisis de forma reciente / expectativa de goles.
- `ingesta.py` — partidos del día, contraste modelo vs realidad, reentrenamiento.
- `scrape_fifa_ranking.py` — descarga el ranking FIFA a `fifa_ranking.csv`.
- `requirements.txt` — dependencias.
- `results_actualizado_mundial_2026.csv` — datos (jugados + fixtures por jugar).
- `MODELO_1_2.ipynb` — notebook original (misma lógica).

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Se abre en http://localhost:8501

## Pestañas
1. **Simulador Mundial** — corre N simulaciones del torneo y muestra
   probabilidad de clasificar / octavos / cuartos / semis / final / campeón.
   Descarga `probabilidades_mundial.csv`.
2. **Predecir partido** — elige dos selecciones y ve la predicción de cada
   método por separado (Poisson+MC, XGBoost sin calibrar, XGBoost calibrado,
   Dixon-Coles) y el ensemble, más los marcadores más probables.
3. **Análisis de forma** — expectativa de goles a partir de los últimos 12
   partidos oficiales de cada selección: decaimiento temporal, ponderación
   por nivel del rival, ancla al ranking FIFA, historial directo (H2H) y
   corrección de resultados bajos. Muestra el desglose paso a paso.
4. **Resultados del día** — ingesta de los partidos de la jornada (manual o
   con el botón "traer de ESPN"), **contraste** de lo modelado vs el resultado
   real (acierto/Brier/LogLoss acumulados) y **reentrenamiento** con los
   partidos nuevos. ⚠️ El almacenamiento en Cloud es efímero: descarga el CSV
   actualizado para conservar los cambios (re-súbelo al repo).
5. **Ranking ELO** — clasificación ELO (historia completa, 8 años).
6. **Evaluación** — comparativa LogLoss / Brier / RPS / Accuracy de cada
   método, con **validación temporal walk-forward** (robustez fuera de muestra).

### Guardar resultados directo a GitHub (opcional, persistente)
La pestaña "Resultados del día" puede commitear el dataset actualizado de
vuelta al repo (así el reentrenamiento es permanente, no solo de sesión).
Para habilitarlo, en Streamlit Cloud → app → Settings → **Secrets**:
```toml
[github]
token    = "github_pat_xxx"   # fine-grained PAT, repo Axewc/Modelo_mundial, Contents: R/W
repo     = "Axewc/Modelo_mundial"
branch   = "main"
path     = "results_actualizado_mundial_2026.csv"
password = "una-clave-secreta" # protege el botón (la app es pública)
```
El token y la clave NUNCA van al repo (`.streamlit/secrets.toml` está en
`.gitignore`). Sin esta config, el botón se oculta y queda la descarga manual.
Hacer commit a `main` dispara un redeploy (~1-2 min) que reentrena el modelo.

### Cuotas de casas de apuestas (comparativo + blend)
En la pestaña **Predecir partido** se puede comparar la probabilidad del
modelo con la **implícita del mercado** (quitando el margen de la casa) y
mezclarlas (blend, pool logarítmico). Las cuotas se traen de The Odds API;
para habilitarlo, en Streamlit Cloud → Secrets:
```toml
[odds]
api_key  = "tu_api_key_de_the_odds_api"
sport    = "soccer_fifa_world_cup"
provider = "the_odds_api"
```
Sin `api_key`, se pueden **ingresar las cuotas 1/X/2 a mano** (no requiere red
ni clave). El blend con peso al mercado mejora la calibración pero hace que el
modelo "siga" al mercado; úsalo según tu objetivo.

### Ranking FIFA (ancla del análisis de forma)
La pestaña de análisis de forma usa el ranking FIFA como ancla. Para
generarlo (una vez, o cuando FIFA publique uno nuevo):
```bash
pip install requests lxml
python scrape_fifa_ranking.py   # crea fifa_ranking.csv → súbelo al repo
```
Si `fifa_ranking.csv` no existe, la app cae automáticamente al **ELO** como
ancla (no se rompe nada).

## Actualizar resultados (ingesta diaria)
1. Escribe el marcador del partido jugado en
   `results_actualizado_mundial_2026.csv` (rellena `home_score`/`away_score`).
2. En la app, pulsa el menú ⋮ → **Rerun** (o `C` para limpiar caché).
   El modelo se reentrena solo con los datos nuevos.

## Desplegar en Streamlit Community Cloud (gratis)
1. Sube esta carpeta a un repositorio de GitHub (incluye el CSV).
2. En https://share.streamlit.io → New app → elige el repo y `app.py`.
3. Deploy. (El primer arranque tarda porque entrena los modelos; luego
   queda en caché.)

## Parámetros ajustables (en `modelo_core.py`)
- `TAU_DAYS = 2920` — ventana de recencia (8 años). Más alto = más peso a la
  historia/reputación; más bajo = más forma reciente.
- En `app.py`, el slider de **Simulaciones** controla la precisión vs. tiempo.

## Notas
- Si `xgboost` no está instalado, la app sigue funcionando con los demás
  métodos (lo avisa en la barra lateral).
- El cuadro de eliminatorias usa la estructura oficial FIFA 2026 (partidos
  73–104) y la tabla de elegibilidad de los 8 mejores terceros.
