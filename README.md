# Modelo predictivo — Mundial 2026

App de Streamlit que ejecuta el pipeline completo (ELO, Poisson, XGBoost,
Dixon-Coles) y simula el torneo con el **cuadro oficial FIFA 2026**.

## Archivos
- `app.py` — interfaz Streamlit (5 pestañas).
- `modelo_core.py` — todo el modelo (ingesta, ELO, Dixon-Coles, simulador).
- `analisis_forma.py` — análisis de forma reciente / expectativa de goles.
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
4. **Ranking ELO** — clasificación ELO (historia completa, 8 años).
5. **Evaluación** — comparativa LogLoss / Brier / RPS / Accuracy de cada
   método, con **validación temporal walk-forward** (robustez fuera de muestra).

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
