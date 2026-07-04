# Backtest point-in-time — Mundiales

Generado por `python backtest_mundiales.py`. Protocolo: entrenamiento solo con partidos anteriores al torneo, ratings congelados al inicio, misma receta que producción (ventana 8 años). El marcador incluye prórroga pero no penales.

## Mundial 2018 — 64 partidos (corte 2018-06-14)

| Modelo               |    RPS |   LogLoss |   Brier |   Accuracy |
|:---------------------|-------:|----------:|--------:|-----------:|
| Dixon-Coles          | 0.205  |    0.971  |  0.578  |      0.594 |
| Ensemble (DC+ELO)    | 0.2102 |    0.9815 |  0.5869 |      0.562 |
| ELO logístico        | 0.2227 |    1.0178 |  0.612  |      0.516 |
| Baseline uniforme    | 0.2439 |    1.0986 |  0.6667 |      0.406 |
| Baseline frecuencias | 0.2506 |    1.0899 |  0.6642 |      0.391 |

## Mundial 2022 — 64 partidos (corte 2022-11-20)

| Modelo               |    RPS |   LogLoss |   Brier |   Accuracy |
|:---------------------|-------:|----------:|--------:|-----------:|
| Ensemble (DC+ELO)    | 0.2119 |    1.0227 |  0.6028 |      0.531 |
| Dixon-Coles          | 0.214  |    1.0327 |  0.6076 |      0.531 |
| ELO logístico        | 0.2191 |    1.0438 |  0.6183 |      0.531 |
| Baseline frecuencias | 0.2349 |    1.0718 |  0.6493 |      0.438 |
| Baseline uniforme    | 0.2387 |    1.0986 |  0.6667 |      0.328 |

## Mundial 2026 (en curso) — 85 partidos (corte 2026-06-11)

| Modelo               |    RPS |   LogLoss |   Brier |   Accuracy |
|:---------------------|-------:|----------:|--------:|-----------:|
| Dixon-Coles          | 0.1598 |    0.863  |  0.5104 |      0.647 |
| Ensemble (DC+ELO)    | 0.1656 |    0.881  |  0.524  |      0.6   |
| ELO logístico        | 0.1813 |    0.9302 |  0.5594 |      0.565 |
| Baseline frecuencias | 0.2191 |    1.0491 |  0.6309 |      0.494 |
| Baseline uniforme    | 0.2346 |    1.0986 |  0.6667 |      0.247 |

## Walk-forward global 2019–hoy — 7,328 partidos

| Modelo                     |    RPS |   LogLoss |   Brier |   Accuracy |
|:---------------------------|-------:|----------:|--------:|-----------:|
| Dixon-Coles (walk-forward) | 0.1682 |    0.8618 |  0.5056 |      0.609 |
| Baseline uniforme          | 0.2398 |    1.0986 |  0.6667 |      0.292 |

![Calibración](img/calibracion.png)
