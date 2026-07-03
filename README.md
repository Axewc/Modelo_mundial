# Modelo predictivo — Mundial 2026

App de Streamlit que ejecuta el pipeline completo (ELO, Poisson, XGBoost,
Dixon-Coles) y simula el torneo con el **cuadro oficial FIFA 2026**.

> *(English)* Probabilistic forecasting system for the FIFA World Cup:
> ELO + Dixon-Coles + calibrated gradient boosting, Monte Carlo tournament
> simulation, and a live Streamlit app. Validated with point-in-time
> backtests on the 2018 and 2022 World Cups (details below).

## 📊 Resultados — backtest point-in-time en Mundiales

¿Cómo le habría ido a **este mismo modelo** prediciendo Mundiales pasados,
sin mirar el futuro? Protocolo: para cada torneo, el modelo se entrena solo
con partidos **anteriores al partido inaugural** (misma receta que producción,
ventana desplazada) y los ratings quedan **congelados** al inicio. Métrica
principal: **RPS** (Ranked Probability Score, menor = mejor — el estándar
para pronósticos 1X2).

| Modelo | Rusia 2018 | Qatar 2022 | 2026 (en curso)\* |
|---|---|---|---|
| **Ensemble (DC+ELO)** | **0.2201** | **0.2186** | 0.1830 |
| ELO logístico | 0.2227 | 0.2191 | **0.1813** |
| Dixon-Coles | 0.2274 | 0.2311 | 0.2029 |
| Baseline frecuencias | 0.2506 | 0.2349 | 0.2191 |
| Baseline uniforme (⅓,⅓,⅓) | 0.2439 | 0.2387 | 0.2346 |

\* 85 partidos jugados al momento del corte. Tablas completas (LogLoss,
Brier, Accuracy) en [`docs/backtest_mundiales.md`](docs/backtest_mundiales.md);
todo se regenera con `python backtest_mundiales.py`.

**Lectura:** el ensemble bate a los dos baselines sin información en los
tres torneos, de forma consistente. El marcador incluye prórroga pero no
penales (convención del dataset), así que "empate" en eliminatorias
significa definición por tanda.

### Calibración

Diagrama de fiabilidad del modelo de goles con **7,328 predicciones
point-in-time** (walk-forward 2019–hoy, reentrenando con ventana móvil de
8 años y regularización elegida por validación en cada refit). Sobre esa
muestra: RPS **0.197** vs 0.240 del baseline uniforme.

![Calibración del modelo](docs/img/calibracion.png)

**Lectura:** los empates están bien calibrados; en victorias el modelo es
**conservador** — cuando asigna >60% la frecuencia real es mayor (las
probabilidades quedan comprimidas hacia el centro). Es consistente con el
hallazgo del grid de regularización (abajo) y marca la dirección de la
siguiente mejora.

### Limitaciones y hallazgos (documentados a propósito)

- **Probé binomial negativa** para engordar la cola de goleadas: la
  validación eligió dispersión ≈ Poisson y quedó marginalmente peor que
  Dixon-Coles — los goles internacionales son esencialmente equidispersos.
  Se conserva como fila comparativa en la app, no como modelo.
- **Los empates casi no mueven el ELO** por el factor de margen
  `log(dif_goles+1)`: limitación conocida del diseño actual.
- **El grid de regularización del DC se queda corto**: en validación,
  `alpha` por debajo del mínimo del grid actual (0.01) sigue mejorando el
  RPS — primer candidato de mejora del modelo.
- **La comparación contra el mercado** (cuotas de cierre) está implementada
  en `backtest.py` (RPS modelo vs mercado, value betting con Kelly), a la
  espera de un histórico de cuotas para los Mundiales pasados.

## Archivos
- `app.py` — interfaz Streamlit (6 pestañas) + `pages/` (editor del Mundial).
- `modelo_core.py` — todo el modelo (ingesta, ELO, Dixon-Coles, simulador).
- `analisis_forma.py` — análisis de forma reciente / expectativa de goles.
- `ingesta.py` — partidos del día, contraste modelo vs realidad, reentrenamiento.
- `backtest_mundiales.py` — **backtest point-in-time de Mundiales** + calibración.
- `backtest.py` — modelo vs mercado (RPS/CLV/value betting con Kelly).
- `odds.py` — cuotas de casas (implícita sin margen, blend con el modelo).
- `nombres.py` — normalización robusta de nombres de selección.
- `gh_commit.py` — persistencia: commit del CSV a GitHub desde la app.
- `scrape_fifa_ranking.py` — descarga el ranking FIFA a `fifa_ranking.csv`.
- `update_results.py` — sincroniza el dataset con el upstream (martj42).
- `requirements.txt` — dependencias.
- `results_actualizado_mundial_2026.csv` — datos (jugados + fixtures por jugar).
- `MODELO_1_2.ipynb` — notebook original (misma lógica).
- `docs/` — resultados del backtest y figuras (regenerables por script).

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

### Mantener el dataset al día (rellenar resultados)
El CSV proviene del dataset comunitario `martj42/international_results`. Para
refrescarlo con todos los resultados nuevos sin perder los fixtures futuros
del Mundial:
```bash
pip install requests pandas
python update_results.py --dry-run   # muestra qué cambiaría
python update_results.py             # actualiza el CSV → commit y push
```
Conserva las filas sin marcador (fixtures por jugar) y deja que el upstream
corrija/agregue los partidos jugados. Para un partido suelto, también puedes
usar la pestaña "Resultados del día" (manual o "Traer de ESPN").

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
