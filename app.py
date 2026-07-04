"""
app.py — Modelo predictivo Mundial 2026 (Streamlit)
===================================================
Ejecuta:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from mundial import modelo_core as mc
from mundial import analisis_forma as af
from mundial import ingesta as ing
from mundial import odds as odds_mod
from mundial import backtest as bt

st.set_page_config(page_title="Mundial 2026 — Modelo", layout="wide",
                   page_icon="⚽", initial_sidebar_state="expanded")

# --- estilo ligero ---
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Entrenando modelos (ELO · Poisson · XGBoost · Dixon-Coles)…")
def cargar(path, _extra, extra_sig):
    # _extra (DataFrame) no se hashea; extra_sig es la clave de caché
    return mc.run_pipeline(path, extra=_extra)


@st.cache_data(show_spinner="Simulando el torneo…")
def simular(_art, n_sim, seed, data_sig):
    # data_sig invalida el cache si cambian los datos; _art no se hashea
    prob, groups = mc.simulate_tournament(_art, n_sim=n_sim, seed=seed)
    return prob, groups


@st.cache_data(show_spinner=False)
def cargar_ranking_fifa(data_sig):
    return af.cargar_ranking_fifa()


@st.cache_data(show_spinner="Validación temporal (walk-forward)…")
def validacion_temporal(_art, data_sig):
    # cómputo pesado bajo demanda; cacheado por firma de datos
    return mc.temporal_validation(_art["df"], _art["dm"])


@st.cache_data(show_spinner="Evaluando 'Análisis de forma' (point-in-time)…")
def evaluar_forma_cached(_art, _ranking, data_sig, rsig):
    # _art y _ranking no se hashean; la clave es (data_sig, rsig)
    return mc.evaluar_forma(_art, ranking=_ranking)


# ---- helpers de formato ----
def pct_col(label, help=None):
    return st.column_config.ProgressColumn(label, format="%.1f%%",
                                           min_value=0, max_value=100, help=help)


def mostrar_mercados(mk, top, local, visitante):
    """Pinta los mercados agrupados + el marcador estrella de cada resultado."""
    st.markdown("**Mercados agrupados** (capturan las goleadas que ningún "
                "marcador individual refleja)")
    a = st.columns(4)
    a[0].metric("Más de 2.5 goles", f"{mk['over25']*100:.0f}%")
    a[1].metric("Ambos marcan", f"{mk['btts']*100:.0f}%")
    a[2].metric(f"{local} por 2+", f"{mk['local_2plus']*100:.0f}%")
    a[3].metric(f"{visitante} por 2+", f"{mk['visit_2plus']*100:.0f}%")
    b = st.columns(3)
    b[0].metric(f"Doble: {local} o empate", f"{mk['doble_1x']*100:.0f}%")
    b[1].metric(f"Doble: {local} o {visitante}", f"{mk['doble_12']*100:.0f}%")
    b[2].metric(f"Doble: empate o {visitante}", f"{mk['doble_x2']*100:.0f}%")
    st.caption(
        f"Marcador más probable **por resultado** — "
        f"Gana {local}: **{top['local']['marcador']}** ({top['local']['prob']*100:.1f}%) · "
        f"Empate: **{top['empate']['marcador']}** ({top['empate']['prob']*100:.1f}%) · "
        f"Gana {visitante}: **{top['visitante']['marcador']}** ({top['visitante']['prob']*100:.1f}%)")


def _gh_secrets():
    """Lee la config [github] de los Secrets de Streamlit (vacío si no hay)."""
    try:
        return dict(st.secrets.get("github", {}))
    except Exception:
        return {}


def _odds_secrets():
    """Lee la config [odds] de los Secrets de Streamlit (vacío si no hay)."""
    try:
        return dict(st.secrets.get("odds", {}))
    except Exception:
        return {}


def _peso_torneo(torneo):
    t = str(torneo).lower()
    if "world cup" in t and "qual" not in t:
        return 2.0
    if "qualif" in t:
        return 1.5
    if "friendly" in t or "amistoso" in t:
        return 0.7
    return 1.0


def mostrar_goles_recientes(df, local, visitante, n=8):
    """Componente compartido por las dos pestañas predictivas: muestra, lado a
    lado, los goles de los últimos partidos de cada selección. Misma fuente y
    mismo formato en 'Predecir partido' y 'Análisis de forma'."""
    st.markdown("##### ⚽ Goles recientes")
    st.caption(f"Últimos {n} partidos oficiales jugados (el más reciente arriba). "
               "GF = goles a favor · GC = goles en contra · forma = últimos 5 (V/E/D).")
    cols = st.columns(2)
    for col, team in zip(cols, (local, visitante)):
        with col:
            t = af.goles_recientes(df, team, n=n)
            if t.empty:
                st.info(f"Sin partidos recientes para **{team}**.")
                continue
            gf, gc = int(t["GF"].sum()), int(t["GC"].sum())
            forma = " ".join(t["Resultado"].head(5).tolist())
            st.markdown(f"**{team}** — {gf} GF / {gc} GC · forma: `{forma}`")
            st.dataframe(t, width="stretch", hide_index=True,
                         column_config={
                             "GF": st.column_config.NumberColumn(format="%d"),
                             "GC": st.column_config.NumberColumn(format="%d"),
                         })


def tabla_metodos(metodos, local, visitante):
    filas = [{"Método": m,
              f"Gana {local}": round(p[2] * 100, 1),
              "Empate": round(p[1] * 100, 1),
              f"Gana {visitante}": round(p[0] * 100, 1)}
             for m, p in metodos.items()]
    return pd.DataFrame(filas)


# ==========================================================
# CARGA
# ==========================================================
st.sidebar.title("⚽ Mundial 2026")
CSV = st.sidebar.text_input("Archivo de datos", mc.CSV_DEFAULT)

# partidos añadidos en la sesión y ya "aplicados" al modelo (reentrenamiento)
aplicados = st.session_state.get("aplicados", [])
extra_df = ing.a_filas_dataset(aplicados) if aplicados else None
extra_sig = f"n{len(aplicados)}" if aplicados else "base"

try:
    art = cargar(CSV, extra_df, extra_sig)
except FileNotFoundError:
    st.error(f"No se encontró el archivo: **{CSV}**")
    st.stop()

data_sig = f"{len(art['df'])}-{art['df']['date'].max()}-{extra_sig}"
if aplicados:
    st.sidebar.success(f"Modelo reentrenado con {len(aplicados)} partido(s) de la sesión.")

with st.sidebar:
    st.divider()
    st.caption("**Calidad del modelo de goles**")
    st.metric("Regularización DC (α)", art.get("alpha_dc", 0.05),
              help="Elegida por validación (deviance de Poisson).")
    c1, c2 = st.columns(2)
    c1.metric("ρ Dixon-Coles", round(art.get("dc_rho", 0), 3),
              help="Corrección de marcadores bajos / empates.")
    c2.metric("Ventaja local γ", round(art.get("dc_gamma", 0), 3))
    st.metric("MAE goles (L / V)",
              f'{art["mae_home"]:.2f} / {art["mae_away"]:.2f}')
    if not art["has_xgb"]:
        st.warning("xgboost no instalado: se omiten esos métodos.")
    st.divider()
    if st.button("🔄 Recalcular (limpiar caché)"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("Tras actualizar el CSV, pulsa este botón.")

st.title("Modelo predictivo — Copa del Mundo 2026")
st.caption("ELO (8 años) · Dixon-Coles (ataque/defensa) · XGBoost calibrado · "
           "simulador con cuadro oficial FIFA.")

tab1, tab2, tab5, tab6, tab3, tab4, tab7 = st.tabs(
    ["🏆 Simulador del torneo", "🆚 Predecir partido", "🔬 Análisis de forma",
     "📥 Resultados del día", "📊 Ranking ELO", "🧪 Evaluación",
     "💹 Backtest vs mercado"])

# ==========================================================
# TAB 1: SIMULADOR
# ==========================================================
with tab1:
    c = st.columns([1.2, 1, 1, 2])
    n_sim = c[0].select_slider("Simulaciones", [2000, 5000, 10000, 20000], value=10000,
                               help="Más simulaciones = más precisión, más tiempo.")
    seed = c[1].number_input("Semilla", value=42, step=1)
    run = c[2].button("▶ Simular", type="primary", width="stretch")

    if run:
        st.session_state["prob"], st.session_state["groups"] = simular(
            art, int(n_sim), int(seed), data_sig)

    if "prob" in st.session_state:
        prob = st.session_state["prob"]
        top3 = prob.head(3).reset_index(drop=True)
        m = st.columns(3)
        medallas = ["🥇", "🥈", "🥉"]
        for i in range(min(3, len(top3))):
            m[i].metric(f"{medallas[i]} {top3.loc[i, 'PAIS']}",
                        f'{top3.loc[i, "P_CAMPEON"]:.1f}%',
                        help=f'Grupo {top3.loc[i, "GRUPO"]} · ELO {int(top3.loc[i, "ELO"])}')

        st.markdown("##### Probabilidad de ser campeón (top 15)")
        st.bar_chart(prob.head(15).set_index("PAIS")["P_CAMPEON"], height=280)

        st.markdown("##### Tabla completa por ronda")
        st.dataframe(
            prob, width="stretch", height=460, hide_index=True,
            column_config={
                "ELO": st.column_config.NumberColumn(format="%d"),
                "P_Clasifica": pct_col("Clasifica", "Pasa de fase de grupos"),
                "P_Octavos": pct_col("Octavos"),
                "P_Cuartos": pct_col("Cuartos"),
                "P_Semis": pct_col("Semis"),
                "P_Final": pct_col("Final"),
                "P_CAMPEON": pct_col("CAMPEÓN"),
            })
        st.download_button("⬇️ Descargar CSV", prob.to_csv(index=False),
                           "probabilidades_mundial.csv", "text/csv")

        with st.expander("Ver grupos reconstruidos"):
            groups = st.session_state.get("groups", {})
            gc = st.columns(4)
            for i, (g, ts) in enumerate(groups.items()):
                gc[i % 4].markdown(f"**Grupo {g}**\n\n" + "\n".join(f"- {t}" for t in ts))
    else:
        st.info("Pulsa **▶ Simular** para calcular las probabilidades del torneo.")

# ==========================================================
# TAB 2: PREDECIR PARTIDO
# ==========================================================
with tab2:
    equipos = art["equipos_actuales"]["PAIS"].sort_values().tolist()
    c = st.columns(4)
    idx_l = equipos.index("Mexico") if "Mexico" in equipos else 0
    local = c[0].selectbox("Local", equipos, index=idx_l)
    visitante = c[1].selectbox("Visitante", equipos, index=1 if idx_l != 1 else 2)
    neutral = c[2].selectbox("Sede neutral", [("No", 0), ("Sí", 1)],
                             format_func=lambda x: x[0])[1]
    torneo = c[3].selectbox("Torneo", [("Mundial", 2.0), ("Euro/Copa América", 1.8),
                                       ("Eliminatoria", 1.5), ("Nations League", 1.3),
                                       ("Amistoso", 0.7), ("Otro", 1.0)],
                            format_func=lambda x: x[0])[1]

    if local == visitante:
        st.warning("Elige equipos distintos.")
    else:
        r = mc.predict_match(art, local, visitante, neutral, torneo)
        ens = r["metodos"]["ENSEMBLE"]

        st.markdown("##### 🎯 Pronóstico (ensemble recomendado)")
        m = st.columns(3)
        m[0].metric(f"Gana {local}", f"{ens[2]*100:.1f}%")
        m[1].metric("Empate", f"{ens[1]*100:.1f}%")
        m[2].metric(f"Gana {visitante}", f"{ens[0]*100:.1f}%")

        gdc = r["goles"].get("Dixon-Coles")
        if gdc:
            st.caption(f"Goles esperados (Dixon-Coles): **{local} {gdc[0]} – {gdc[1]} {visitante}**")

        st.divider()
        mostrar_goles_recientes(art["df"], local, visitante, n=8)

        st.divider()
        st.markdown("##### Comparativa por método")
        st.dataframe(
            tabla_metodos(r["metodos"], local, visitante),
            width="stretch", hide_index=True,
            column_config={
                f"Gana {local}": pct_col(f"Gana {local}"),
                "Empate": pct_col("Empate"),
                f"Gana {visitante}": pct_col(f"Gana {visitante}"),
            })

        with st.expander("Marcadores más probables (Dixon-Coles)"):
            st.dataframe(r["marcadores"], width="stretch", hide_index=True,
                         column_config={"Prob %": pct_col("Probabilidad")})

        st.divider()
        try:
            Mdc = mc.dc_score_matrix(art["dc_full"], local, visitante, bool(neutral))
            mostrar_mercados(af.mercados_agrupados(Mdc),
                             af.marcador_top_por_resultado(Mdc), local, visitante)
        except Exception:
            pass

        # ---- Mercado (casas de apuestas): comparativo + blend ----
        st.divider()
        st.markdown("##### 🎰 Mercado (casas de apuestas)")
        od = _odds_secrets()
        if od.get("api_key") and st.button("🌐 Traer cuotas (The Odds API)"):
            try:
                st.session_state["cuotas_df"] = odds_mod.traer_cuotas(
                    od["api_key"], od.get("sport", odds_mod.SPORT_DEFAULT))
                st.success(f"{len(st.session_state['cuotas_df'])} partidos con cuotas.")
            except Exception as e:
                st.warning(f"No se pudieron traer cuotas ({type(e).__name__}): {e}")

        fila = odds_mod.buscar_partido(st.session_state.get("cuotas_df"), local, visitante)
        with st.expander("Ingresar cuotas a mano (1 / X / 2)",
                         expanded=fila is None):
            oc = st.columns(3)
            ch = oc[0].number_input(f"Cuota {local}", 1.01, 1000.0, 2.00, 0.05, key="oc_h")
            cd = oc[1].number_input("Cuota empate", 1.01, 1000.0, 3.20, 0.05, key="oc_d")
            ca = oc[2].number_input(f"Cuota {visitante}", 1.01, 1000.0, 3.50, 0.05, key="oc_a")
            usar_manual = st.checkbox("Usar estas cuotas", value=fila is None)

        market = None
        if fila is not None and not usar_manual:
            # reorientar a [away=visitante, draw, home=local] segun el orden de la fila
            if fila["home_team"] == odds_mod.normaliza_equipo(local):
                market = np.array([fila["p_away"], fila["p_draw"], fila["p_home"]])
            else:
                market = np.array([fila["p_home"], fila["p_draw"], fila["p_away"]])
            st.caption(f"Cuotas de **{int(fila['n_casas'])} casas** · margen "
                       f"{fila['margen']}% · {fila['home_team']} {fila['cuota_home']} / "
                       f"X {fila['cuota_draw']} / {fila['away_team']} {fila['cuota_away']}")
        elif usar_manual:
            try:
                market = odds_mod.implied_probs(ch, cd, ca)  # [away,draw,home], home=local
            except ValueError as e:
                st.warning(str(e))

        if market is not None:
            w = st.slider("Peso del mercado en el blend", 0.0, 1.0, 0.5, 0.05,
                          help="0 = solo modelo · 1 = solo mercado (pool logarítmico).")
            mezcla = odds_mod.blend(ens, market, w)
            comp = pd.DataFrame({
                "Fuente": ["Modelo", "Mercado", f"Blend (w={w:.2f})"],
                f"Gana {local}": [ens[2], market[2], mezcla[2]],
                "Empate": [ens[1], market[1], mezcla[1]],
                f"Gana {visitante}": [ens[0], market[0], mezcla[0]],
            })
            for c in comp.columns[1:]:
                comp[c] = (comp[c] * 100).round(1)
            st.dataframe(comp, width="stretch", hide_index=True)
            # "valor": el modelo da MÁS probabilidad que el mercado
            val = []
            for et, lbl in [(2, f"Gana {local}"), (1, "Empate"), (0, f"Gana {visitante}")]:
                d = (ens[et] - market[et]) * 100
                if d > 2:
                    val.append(f"**{lbl}** (+{d:.1f} pts vs mercado)")
            st.caption("💡 Valor del modelo (donde ve más probabilidad que el mercado): "
                       + (", ".join(val) if val else "ninguno destacado."))

        # ---- Eliminatoria: quién avanza (prórroga + penales) ----
        st.divider()
        ko = st.checkbox("🏆 Eliminatoria (a un partido): ¿quién avanza?")
        if ko:
            lh, la = art["dc_full"]["lambdas"](local, visitante, bool(neutral))
            fav = st.checkbox("Ligera ventaja al favorito en penales (si no, 50/50)")
            shoot = 0.5
            if fav and (lh + la) > 0:
                shoot = float(np.clip(0.5 + 0.25 * (lh - la) / (lh + la), 0.35, 0.65))
            av = mc.prob_avance(ens, lh, la, shootout=shoot)
            cA, cB = st.columns(2)
            cA.metric(f"🏆 Avanza {local}", f"{av[1]*100:.1f}%")
            cB.metric(f"🏆 Avanza {visitante}", f"{av[0]*100:.1f}%")
            st.caption(
                f"El empate de los 90' ({ens[1]*100:.0f}%) se resuelve con "
                f"**prórroga** (goles esperados ×1/3) y, si persiste, **penales** "
                f"({'ventaja '+str(round(shoot*100))+'% al local' if fav else '50/50'}). "
                "Es el mismo criterio que usa el simulador del torneo.")

# ==========================================================
# TAB 5: ANÁLISIS DE FORMA
# ==========================================================
with tab5:
    st.markdown("##### Expectativa de goles por forma reciente")
    st.caption(
        "Construye el pronóstico desde los **últimos 12 partidos oficiales** "
        "de cada selección: 1) decaimiento temporal (lo nuevo pesa más), "
        "2) ponderación por nivel del rival, 3) ancla al ranking, "
        "4) historial directo (H2H) y 5) corrección de resultados bajos.")

    ranking = cargar_ranking_fifa(data_sig)
    if ranking is None:
        st.warning(
            "No se encontró **fifa_ranking.csv** → se usa el **ELO** como ancla. "
            "Para usar el ranking FIFA oficial, ejecuta `python scrape_fifa_ranking.py` "
            "localmente y sube el CSV al repo.")

    dff = art["df"]
    eqs = sorted(pd.unique(pd.concat([dff["home_team"], dff["away_team"]])).tolist())
    cc = st.columns(4)
    il = eqs.index("Mexico") if "Mexico" in eqs else 0
    loc = cc[0].selectbox("Local", eqs, index=il, key="f_loc")
    vis = cc[1].selectbox("Visitante", eqs, index=1 if il != 1 else 2, key="f_vis")
    neu = cc[2].selectbox("Sede neutral", [("No", False), ("Sí", True)],
                          format_func=lambda x: x[0], key="f_neu")[1]
    nmatches = cc[3].slider("Partidos a considerar", 6, 20, af.N_PARTIDOS, key="f_n")

    if loc == vis:
        st.warning("Elige selecciones distintas.")
    else:
        r = af.predecir_partido_forma(dff, art["elo"], ranking, loc, vis, neu, n=nmatches)
        p, g = r["prob"], r["goles_esperados"]
        st.caption(f"Ancla usada: **{r['base_nivel']}** · "
                   f"índice de nivel — {loc}: {r['indices']['local']} · "
                   f"{vis}: {r['indices']['visitante']}")

        m = st.columns(3)
        m[0].metric(f"Gana {loc}", f"{p['local']*100:.1f}%")
        m[1].metric("Empate", f"{p['empate']*100:.1f}%")
        m[2].metric(f"Gana {vis}", f"{p['visitante']*100:.1f}%")
        st.caption(f"⚽ Goles esperados: **{loc} {g['local']} – {g['visitante']} {vis}**"
                   + (f" · H2H (últimos {r['h2h']['n']}): dif. {r['h2h']['diff']:+.2f} "
                      f"a favor de {loc if r['h2h']['diff']>=0 else vis}"
                      if r["h2h"]["n"] else " · sin historial directo"))

        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown("**Marcadores más probables**")
            st.dataframe(r["marcadores"], width="stretch", hide_index=True,
                         column_config={"Prob %": pct_col("Probabilidad")})
        with c2:
            st.markdown("**Forma reciente (ataque / defensa esperados)**")
            fl, fv = r["forma_local"], r["forma_visitante"]
            st.dataframe(pd.DataFrame([
                {"Selección": loc, "Ataque": round(fl["ataque"], 2),
                 "Defensa": round(fl["defensa"], 2), "Partidos": fl["n"]},
                {"Selección": vis, "Ataque": round(fv["ataque"], 2),
                 "Defensa": round(fv["defensa"], 2), "Partidos": fv["n"]},
            ]), width="stretch", hide_index=True)
            st.caption("Ataque: goles esperados que marca · Defensa: goles que "
                       "recibe, ya ajustados por el nivel de los rivales.")

        st.divider()
        mostrar_goles_recientes(dff, loc, vis, n=nmatches)

        st.divider()
        mostrar_mercados(r["mercados"], r["top_resultado"], loc, vis)

        with st.expander(f"Ver últimos partidos ponderados — {loc}"):
            st.dataframe(r["forma_local"]["detalle"], width="stretch",
                         hide_index=True)
        with st.expander(f"Ver últimos partidos ponderados — {vis}"):
            st.dataframe(r["forma_visitante"]["detalle"], width="stretch",
                         hide_index=True)


# ==========================================================
# TAB 6: RESULTADOS DEL DÍA (ingesta + contraste + reentrenar)
# ==========================================================
with tab6:
    st.markdown("##### 📥 Resultados del día — ingesta, contraste y reentrenamiento")
    st.caption("Trae o añade los partidos del día, guarda la predicción del "
               "modelo, registra los resultados reales para contrastar, y "
               "reentrena con los nuevos partidos. ⚠️ En Streamlit Cloud el "
               "almacenamiento es **efímero**: usa el botón de descarga para "
               "conservar los cambios (re-sube el CSV al repo).")

    pj_all = art["por_jugar"].copy()
    pj_all["date"] = pd.to_datetime(pj_all["date"]).dt.strftime("%Y-%m-%d")
    default_day = (pd.to_datetime(pj_all["date"]).min()
                   if len(pj_all) else pd.Timestamp("today"))
    fecha = st.date_input("Día", value=default_day)
    fsel = pd.Timestamp(fecha).strftime("%Y-%m-%d")

    cbtn = st.columns([1.3, 1, 2])
    if cbtn[0].button("🌐 Traer de ESPN (Mundial)"):
        try:
            fx = ing.traer_partidos_del_dia(fsel)
            st.session_state["fx_dia"] = fx.to_dict("records")
            st.success(f"Traídos {len(fx)} partido(s). Revísalos y edítalos abajo.")
        except Exception as e:
            st.warning(f"No se pudo traer automáticamente ({type(e).__name__}). "
                       f"Añade los partidos a mano en la tabla. Detalle: {e}")

    # Tabla editable: fixtures del CSV de ese día + lo traído de ESPN.
    pj_dia = pj_all[pj_all["date"] == fsel]
    traidos = pd.DataFrame(st.session_state.get("fx_dia", []))
    partes = [d for d in [pj_dia, traidos] if len(d)]
    tabla = (pd.concat(partes, ignore_index=True).reindex(columns=ing.COLS)
             if partes else pd.DataFrame(columns=ing.COLS))
    if len(tabla):
        # dedup por partido prefiriendo la fila que YA trae marcador (p.ej. la
        # de ESPN sobre el fixture vacío del CSV); índice limpio para mostrar.
        tabla["_con_marcador"] = tabla[["home_score", "away_score"]].notna().any(axis=1)
        tabla = (tabla.sort_values("_con_marcador")
                 .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
                 .drop(columns="_con_marcador")
                 .sort_values(["date", "home_team"]).reset_index(drop=True))
        tabla["date"] = tabla["date"].fillna(fsel)
    # Tipos consistentes para el editor (evita choques de column_config):
    # marcadores numéricos y 'neutral' booleano (aunque la tabla esté vacía).
    for c in ("home_score", "away_score"):
        tabla[c] = pd.to_numeric(tabla.get(c), errors="coerce")
    tabla["neutral"] = (tabla.get("neutral")
                        .map(lambda v: str(v).strip().lower() in ("true", "1", "sí", "si"))
                        .astype(bool))

    st.caption("Edita marcadores y añade filas (botón ➕). Deja el marcador "
               "vacío si el partido aún no se juega.")
    editado = st.data_editor(
        tabla, num_rows="dynamic", width="stretch", key="editor_dia",
        column_config={
            "home_score": st.column_config.NumberColumn("GL", min_value=0, step=1),
            "away_score": st.column_config.NumberColumn("GV", min_value=0, step=1),
            "neutral": st.column_config.CheckboxColumn("Neutral"),
        })

    c2 = st.columns(2)
    if c2[0].button("📸 Guardar predicción del modelo"):
        log = st.session_state.get("pred_log", [])
        idx = {(r["date"], r["home_team"], r["away_team"]): i
               for i, r in enumerate(log)}
        nuevos = 0
        for _, row in editado.iterrows():
            if not row.get("home_team") or not row.get("away_team"):
                continue
            try:
                neu = 1 if str(row.get("neutral")).lower() in ("true", "1", "sí", "si") else 0
                p = mc.predict_match(art, row["home_team"], row["away_team"],
                                     neu, _peso_torneo(row.get("tournament")))
                p = p["metodos"]["ENSEMBLE"]
            except Exception:
                continue  # equipo no presente en los datos: se omite
            key = (str(row["date"]), row["home_team"], row["away_team"])
            entry = {"date": str(row["date"]), "home_team": row["home_team"],
                     "away_team": row["away_team"],
                     "p_away": float(p[0]), "p_draw": float(p[1]), "p_home": float(p[2]),
                     "res": ing.resultado_clase(row.get("home_score"), row.get("away_score"))}
            if key in idx:
                log[idx[key]].update(entry)
            else:
                log.append(entry)
                nuevos += 1
        st.session_state["pred_log"] = log
        st.success(f"Predicciones guardadas ({nuevos} nuevas · {len(log)} en total).")

    if c2[1].button("✅ Registrar resultados y reentrenar", type="primary"):
        comp = editado[editado["home_score"].notna() & editado["away_score"].notna()]
        log = st.session_state.get("pred_log", [])
        idx = {(r["date"], r["home_team"], r["away_team"]): i
               for i, r in enumerate(log)}
        apl = st.session_state.get("aplicados", [])
        seen = {(str(a["date"]), a["home_team"], a["away_team"]) for a in apl}
        add = 0
        for _, row in comp.iterrows():
            key = (str(row["date"]), row["home_team"], row["away_team"])
            res = ing.resultado_clase(row["home_score"], row["away_score"])
            if key in idx:
                log[idx[key]]["res"] = res
            if key not in seen:
                apl.append({c: row.get(c) for c in ing.COLS})
                seen.add(key)
                add += 1
        st.session_state["pred_log"] = log
        st.session_state["aplicados"] = apl
        if add:
            st.success(f"{add} resultado(s) añadidos. Reentrenando el modelo…")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.info("No hay partidos con marcador completo para añadir.")

    # ---- Contraste modelo vs realidad ----
    st.divider()
    st.markdown("##### 📊 Contraste: lo modelado vs lo que pasó")
    log_df = pd.DataFrame(st.session_state.get("pred_log", []))
    detalle, resumen = (ing.evalua_contraste(log_df) if len(log_df)
                        else (pd.DataFrame(), {}))
    if resumen:
        m = st.columns(4)
        m[0].metric("Partidos resueltos", resumen["n"])
        m[1].metric("Acierto 1X2", f"{resumen['accuracy']*100:.1f}%")
        m[2].metric("Brier", f"{resumen['brier']:.3f}")
        m[3].metric("LogLoss", f"{resumen['logloss']:.3f}",
                    help=f"Baseline 33/33/33 = {resumen['logloss_baseline']:.3f} "
                         "(menor que esto = el modelo aporta).")
        show = detalle.copy()
        cmap = {"p_away": "V", "p_draw": "E", "p_home": "L"}
        show["pred"] = show[["p_away", "p_draw", "p_home"]].idxmax(axis=1).map(cmap)
        show["real"] = show["res"].map({0: "V", 1: "E", 2: "L"})
        show["✓"] = show["acierto"].map({1: "✅", 0: "❌"})
        st.dataframe(
            show[["date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                  "pred", "real", "✓", "brier", "logloss"]].round(3),
            width="stretch", hide_index=True)
        st.download_button("⬇️ Descargar track-record (CSV)",
                           log_df.to_csv(index=False), "track_record.csv", "text/csv")
    else:
        st.info("Aún no hay predicciones resueltas. Guarda la predicción del "
                "modelo y luego registra los resultados reales.")

    # ---- Estado de sesión + exportar dataset ----
    st.divider()
    apl = st.session_state.get("aplicados", [])
    st.caption(f"Partidos de la sesión aplicados al modelo: **{len(apl)}** "
               "(el modelo ya está reentrenado con ellos).")
    if apl:
        full = ing.merge_resultados(pd.read_csv(CSV), ing.a_filas_dataset(apl))
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Descargar dataset actualizado (súbelo al repo)",
                           full.to_csv(index=False),
                           "results_actualizado_mundial_2026.csv", "text/csv")
        if d2.button("🗑️ Descartar partidos de la sesión"):
            st.session_state["aplicados"] = []
            st.cache_resource.clear()
            st.rerun()

    # ---- Commit directo a GitHub (persistente) ----
    st.divider()
    st.markdown("##### 💾 Guardar en GitHub (persistente)")
    gh = _gh_secrets()
    if not gh.get("token"):
        st.info("No configurado. Añade una sección `[github]` (token, repo, "
                "branch, path, password) en los **Secrets** de Streamlit para "
                "habilitar el commit directo. Mientras tanto, usa la descarga "
                "de arriba.")
    elif not apl:
        st.caption("No hay partidos de la sesión para subir todavía.")
    else:
        st.caption("Sube el dataset actualizado al repo (dispara redeploy y deja "
                   "el modelo reentrenado de forma permanente).")
        pwd = st.text_input("Contraseña", type="password", key="gh_pwd")
        tr_too = st.checkbox("Subir también el track-record", value=True)
        if st.button("💾 Commit a GitHub", type="primary"):
            import hmac
            if not hmac.compare_digest(str(pwd), str(gh.get("password", ""))):
                st.error("Contraseña incorrecta.")
            else:
                try:
                    from mundial import gh_commit
                    full = ing.merge_resultados(pd.read_csv(CSV),
                                                ing.a_filas_dataset(apl))
                    branch = gh.get("branch", "main")
                    res = gh_commit.commit_file(
                        gh["token"], gh["repo"],
                        gh.get("path", "results_actualizado_mundial_2026.csv"),
                        full.to_csv(index=False),
                        f"Resultados vía app: +{len(apl)} partido(s)", branch=branch)
                    extra = ""
                    if tr_too and st.session_state.get("pred_log"):
                        tr_csv = pd.DataFrame(st.session_state["pred_log"]).to_csv(index=False)
                        gh_commit.commit_file(gh["token"], gh["repo"], "track_record.csv",
                                              tr_csv, "Track-record vía app", branch=branch)
                        extra = " · track-record subido"
                    url = res.get("html_url") or "(commit ok)"
                    st.success(f"✅ Commit hecho{extra}. La app se redesplegará en "
                               f"~1-2 min y quedará reentrenada.\n\n{url}")
                except Exception as e:
                    st.error(f"Falló el commit: {type(e).__name__}: {e}")


# ==========================================================
# TAB 3: RANKING ELO
# ==========================================================
with tab3:
    st.markdown("##### Ranking ELO (historia completa · ventana 8 años)")
    rk = art["ranking"].copy()
    rk.index = rk.index + 1
    st.dataframe(rk, width="stretch", height=620,
                 column_config={"ELO": st.column_config.ProgressColumn(
                     "ELO", format="%.0f",
                     min_value=float(rk["ELO"].min()), max_value=float(rk["ELO"].max()))})

# ==========================================================
# TAB 4: EVALUACIÓN
# ==========================================================
with tab4:
    st.markdown("##### Comparativa de métodos en TEST")
    st.caption("**LogLoss, Brier, RPS:** menor es mejor · **Accuracy:** mayor es mejor. "
               "RPS es la métrica estándar para pronósticos 1X2.")
    st.caption(f"_Neg-Binomial_ = variante de Dixon-Coles con cola más gruesa "
               f"(sobredispersión, r={art.get('r_nb', '—')}). Se incluye como "
               "comparación; si no mejora a Dixon-Coles, no se adopta.")
    te = art["tabla_eval"]
    sty = (te.style
           .highlight_min(subset=["LogLoss", "Brier", "RPS"], color="#1b5e20")
           .highlight_max(subset=["Accuracy"], color="#1b5e20")
           .format({"LogLoss": "{:.4f}", "Brier": "{:.4f}",
                    "RPS": "{:.4f}", "Accuracy": "{:.3f}"}))
    st.dataframe(sty, width="stretch", hide_index=True)
    st.caption("El **mejor** valor de cada columna aparece resaltado en verde.")

    st.divider()
    st.markdown("##### ¿'Análisis de forma' le gana a Dixon-Coles?")
    st.caption("Mete el modelo de la pestaña 🔬 en el **mismo** test (point-in-time) "
               "para que decidan los datos, no la autoridad. El consenso "
               "(Dixon-Coles, 1997 + literatura) favorece el modelo ajustado por "
               "verosimilitud; aquí se confirma —o no— en TUS datos. "
               "_Cómputo bajo demanda (~1-2 min)._")
    if st.button("▶ Evaluar 'Análisis de forma'"):
        rk = cargar_ranking_fifa(data_sig)
        rsig = f"fifa{len(rk) if rk else 0}"
        st.session_state["forma_eval"] = evaluar_forma_cached(art, rk, data_sig, rsig)

    fe = st.session_state.get("forma_eval")
    if fe is not None:
        cols = ["Modelo", "LogLoss", "Brier", "RPS", "Accuracy"]
        comp = (pd.concat([art["tabla_eval"][cols],
                           pd.DataFrame([{k: fe[k] for k in cols}])],
                          ignore_index=True)
                .sort_values("RPS").reset_index(drop=True))
        styc = (comp.style
                .highlight_min(subset=["LogLoss", "Brier", "RPS"], color="#1b5e20")
                .highlight_max(subset=["Accuracy"], color="#1b5e20")
                .format({"LogLoss": "{:.4f}", "Brier": "{:.4f}",
                         "RPS": "{:.4f}", "Accuracy": "{:.3f}"}))
        st.dataframe(styc, width="stretch", hide_index=True)

        msk = art["tabla_eval"]["Modelo"] == "Dixon-Coles"
        if msk.any():
            dc_rps = float(art["tabla_eval"].loc[msk, "RPS"].iloc[0])
            f_rps = float(fe["RPS"])
            st.info((f"✅ 'Análisis de forma' (RPS {f_rps:.4f}) **mejora** a "
                     f"Dixon-Coles (RPS {dc_rps:.4f}) en tus datos."
                     if f_rps < dc_rps else
                     f"⚠️ Dixon-Coles (RPS {dc_rps:.4f}) **le gana** a 'Análisis de "
                     f"forma' (RPS {f_rps:.4f}), en línea con el consenso.")
                    + f"  · {fe.get('n_test', '?')} partidos de test.")
        st.caption("La forma recibe ranking FIFA y ELO **actuales** (leak menor que "
                   "la favorece); si aun así pierde, la conclusión es robusta.")

    st.divider()
    st.markdown("##### Validación temporal (walk-forward)")
    st.caption("Ventana expansiva en el tiempo: se entrena con el pasado y se "
               "evalúa en el bloque siguiente, repetido en varios *folds*. "
               "**LogLoss/RPS/Brier:** menor es mejor · **Accuracy:** mayor es mejor. "
               "Una desviación (`_sd`) baja entre folds indica un modelo estable "
               "en el tiempo y poco sobreajustado. _Cómputo pesado: se calcula "
               "bajo demanda para no ralentizar el arranque._")

    if st.button("▶ Calcular validación temporal"):
        st.session_state["temporal"] = validacion_temporal(art, data_sig)

    res = st.session_state.get("temporal")
    if res is not None and not res["resumen"].empty:
        tt = res["resumen"]
        styt = (tt.style
                .highlight_min(subset=["LogLoss", "RPS", "Brier"], color="#1b5e20")
                .highlight_max(subset=["Accuracy"], color="#1b5e20")
                .format({"LogLoss": "{:.4f}", "LogLoss_sd": "{:.4f}",
                         "RPS": "{:.4f}", "RPS_sd": "{:.4f}",
                         "Brier": "{:.4f}", "Accuracy": "{:.3f}"}))
        st.dataframe(styt, width="stretch", hide_index=True)
        folds = res["folds"]
        if folds is not None and not folds.empty:
            with st.expander("Ver métricas por fold"):
                st.dataframe(folds, width="stretch", hide_index=True)
    else:
        st.info("Pulsa **▶ Calcular validación temporal** para ejecutar el "
                "walk-forward (tarda unos segundos).")

# ==========================================================
# TAB 7: BACKTEST VS MERCADO
# ==========================================================
with tab7:
    st.markdown("##### ¿Le gana el modelo al mercado?")
    st.caption("Contrasta el Dixon-Coles (**point-in-time**, walk-forward: cada "
               "partido se predice solo con datos anteriores) contra las "
               "**cuotas de cierre** sobre los mismos partidos. Es el único "
               "benchmark que de verdad importa para un ángulo de trading: si el "
               "mercado gana, aún no hay edge propio. Como se apuesta *a la línea "
               "de cierre*, un ROI>0 ya es una señal fuerte.")
    st.caption("Necesitas un CSV de cuotas históricas con columnas: "
               "`date, home_team, away_team, cuota_home, cuota_draw, cuota_away` "
               "(decimales). Fuente típica para clubes: football-data.co.uk.")

    up = st.file_uploader("CSV de cuotas (1X2 decimales)", type=["csv"], key="bt_odds")
    cc = st.columns(4)
    edge_min = cc[0].slider("Edge mínimo", 0.0, 0.15, 0.03, 0.01,
                            help="Apuesta solo si el modelo da al menos esta "
                                 "ventaja sobre el mercado.")
    kelly = cc[1].slider("Fracción Kelly", 0.0, 1.0, 0.0, 0.1,
                         help="0 = staking plano (1 unidad por apuesta).")
    retrain = cc[2].select_slider("Reentrenar cada", [5, 10, 15, 30, 50], value=15,
                                  help="Cada cuántos partidos se reentrena el "
                                       "Dixon-Coles en el walk-forward.")
    min_train = cc[3].number_input("Mín. partidos de entrenamiento", 100, 6000, 800, 100)

    if up is not None and st.button("▶ Correr backtest", type="primary"):
        try:
            odds_df = pd.read_csv(up)
            with st.spinner("Walk-forward + scoring…"):
                bk = bt.construir_backtest(art, odds_df, retrain_every=int(retrain),
                                           min_train=int(min_train))
            st.session_state["bk"] = bk
        except Exception as e:
            st.error(f"Error en el backtest: {e}")

    bk = st.session_state.get("bk")
    if bk is not None:
        n = bk["n"]
        if n == 0:
            st.warning("No se cruzó ningún partido jugado con cuotas. Revisa que "
                       "los nombres de equipo y las fechas del CSV de cuotas casen "
                       "con el dataset (mismo idioma/ortografía).")
        else:
            tab_cmp, winrate = bt.comparar_modelo_mercado(
                bk["model_P"], bk["market_P"], bk["outcomes"])
            rps_mod = float(tab_cmp.loc[tab_cmp["Quién"] == "Modelo", "RPS"].iloc[0])
            rps_mkt = float(tab_cmp.loc[tab_cmp["Quién"] == "Mercado", "RPS"].iloc[0])

            m = st.columns(3)
            m[0].metric("Partidos con cuota", n)
            m[1].metric("RPS modelo", f"{rps_mod:.4f}",
                        delta=f"{rps_mod - rps_mkt:+.4f} vs mercado",
                        delta_color="inverse")
            m[2].metric("Modelo gana al mercado", f"{winrate*100:.0f}%",
                        help="Partidos donde el RPS del modelo es mejor que el del mercado.")

            st.info("✅ El modelo iguala o supera al mercado en RPS."
                    if rps_mod <= rps_mkt else
                    "⚠️ El mercado gana en RPS: aún no hay edge agregado claro "
                    "(esto es lo normal; batir al mercado es difícil).")
            st.dataframe(tab_cmp, hide_index=True, width="stretch",
                         column_config={
                             "RPS": st.column_config.NumberColumn(format="%.4f"),
                             "LogLoss": st.column_config.NumberColumn(format="%.4f"),
                             "Brier": st.column_config.NumberColumn(format="%.4f")})

            st.markdown("##### Apuestas de valor (a cuotas de cierre)")
            det, res = bt.value_bets(bk["model_P"], bk["market_P"], bk["odds"],
                                     bk["outcomes"], edge_min=edge_min, kelly_frac=kelly)
            r = st.columns(4)
            r[0].metric("Apuestas", res["n_apuestas"])
            r[1].metric("Acierto", f"{res['hit_rate']*100:.0f}%")
            r[2].metric("ROI", f"{res['ROI']*100:+.1f}%",
                        help="A cuotas de cierre. ROI>0 ya es señal fuerte de edge.")
            r[3].metric("P&L (unidades)", f"{res['profit']:+.2f}")

            if n < 100:
                st.warning(f"Solo {n} partidos cruzados: muestra pequeña, alta "
                           "incertidumbre. Úsalo como indicio, no como conclusión. "
                           "Para concluir necesitas cientos/miles de partidos con cuota.")
            if not det.empty:
                with st.expander("Ver apuestas"):
                    st.dataframe(det.drop(columns=["i"]), hide_index=True, width="stretch")
    else:
        st.info("Sube un CSV de cuotas y pulsa **▶ Correr backtest**. "
                "Sin cuotas históricas no se puede medir el edge real del modelo.")
