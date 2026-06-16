"""
app.py — Modelo predictivo Mundial 2026 (Streamlit)
===================================================
Ejecuta:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import modelo_core as mc

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
def cargar(path):
    return mc.run_pipeline(path)


@st.cache_data(show_spinner="Simulando el torneo…")
def simular(_art, n_sim, seed, data_sig):
    # data_sig invalida el cache si cambian los datos; _art no se hashea
    prob, groups = mc.simulate_tournament(_art, n_sim=n_sim, seed=seed)
    return prob, groups


# ---- helpers de formato ----
def pct_col(label, help=None):
    return st.column_config.ProgressColumn(label, format="%.1f%%",
                                           min_value=0, max_value=100, help=help)


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

try:
    art = cargar(CSV)
except FileNotFoundError:
    st.error(f"No se encontró el archivo: **{CSV}**")
    st.stop()

data_sig = f"{len(art['df'])}-{art['df']['date'].max()}"

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

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆 Simulador del torneo", "🆚 Predecir partido", "📊 Ranking ELO", "🧪 Evaluación"])

# ==========================================================
# TAB 1: SIMULADOR
# ==========================================================
with tab1:
    c = st.columns([1.2, 1, 1, 2])
    n_sim = c[0].select_slider("Simulaciones", [2000, 5000, 10000, 20000], value=10000,
                               help="Más simulaciones = más precisión, más tiempo.")
    seed = c[1].number_input("Semilla", value=42, step=1)
    run = c[2].button("▶ Simular", type="primary", use_container_width=True)

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
            prob, use_container_width=True, height=460, hide_index=True,
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

        st.markdown("##### Comparativa por método")
        st.dataframe(
            tabla_metodos(r["metodos"], local, visitante),
            use_container_width=True, hide_index=True,
            column_config={
                f"Gana {local}": pct_col(f"Gana {local}"),
                "Empate": pct_col("Empate"),
                f"Gana {visitante}": pct_col(f"Gana {visitante}"),
            })

        with st.expander("Marcadores más probables (Dixon-Coles)"):
            st.dataframe(r["marcadores"], use_container_width=True, hide_index=True,
                         column_config={"Prob %": pct_col("Probabilidad")})

# ==========================================================
# TAB 3: RANKING ELO
# ==========================================================
with tab3:
    st.markdown("##### Ranking ELO (historia completa · ventana 8 años)")
    rk = art["ranking"].copy()
    rk.index = rk.index + 1
    st.dataframe(rk, use_container_width=True, height=620,
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
    te = art["tabla_eval"]
    sty = (te.style
           .highlight_min(subset=["LogLoss", "Brier", "RPS"], color="#1b5e20")
           .highlight_max(subset=["Accuracy"], color="#1b5e20")
           .format({"LogLoss": "{:.4f}", "Brier": "{:.4f}",
                    "RPS": "{:.4f}", "Accuracy": "{:.3f}"}))
    st.dataframe(sty, use_container_width=True, hide_index=True)
    st.caption("El **mejor** valor de cada columna aparece resaltado en verde.")
