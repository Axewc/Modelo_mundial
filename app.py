"""
app.py — Modelo predictivo Mundial 2026 (Streamlit)
===================================================
Ejecuta:  streamlit run app.py

Lee results_actualizado_mundial_2026.csv, corre todo el pipeline (con cache)
y muestra: ranking ELO, predictor de partido (todos los metodos),
simulador del Mundial y la tabla de evaluacion.
"""
import streamlit as st
import pandas as pd
import modelo_core as mc

st.set_page_config(page_title="Mundial 2026 — Modelo", layout="wide")
st.title("⚽ Modelo predictivo — Mundial 2026")


@st.cache_resource(show_spinner="Entrenando modelos (ELO, Poisson, XGBoost, Dixon-Coles)...")
def cargar(path):
    return mc.run_pipeline(path)


@st.cache_data(show_spinner="Simulando el torneo...")
def simular(_art, n_sim, seed):
    # _art no se hashea; el cache depende de (n_sim, seed)
    prob, groups = mc.simulate_tournament(_art, n_sim=n_sim, seed=seed)
    return prob


CSV = st.sidebar.text_input("Archivo de datos", mc.CSV_DEFAULT)

try:
    art = cargar(CSV)
except FileNotFoundError:
    st.error(f"No se encontro el archivo: {CSV}")
    st.stop()

if not art["has_xgb"]:
    st.sidebar.warning("xgboost no esta instalado: se omiten los metodos XGBoost.")

st.sidebar.metric("MAE goles local (Poisson)", round(art["mae_home"], 3))
st.sidebar.metric("MAE goles visitante (Poisson)", round(art["mae_away"], 3))

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆 Simulador Mundial", "🆚 Predecir partido", "📊 Ranking ELO", "🧪 Evaluación"])

# ----------------------------------------------------------
# TAB 1: SIMULADOR
# ----------------------------------------------------------
with tab1:
    st.subheader("Probabilidades del torneo (Monte Carlo)")
    col = st.columns([1, 1, 2])
    n_sim = col[0].select_slider("Simulaciones", [2000, 5000, 10000, 20000], value=10000)
    seed = col[1].number_input("Semilla", value=42, step=1)
    if st.button("Simular torneo", type="primary"):
        st.session_state["prob"] = simular(art, int(n_sim), int(seed))

    if "prob" in st.session_state:
        prob = st.session_state["prob"]
        c1, c2, c3, c4 = st.columns(4)
        top = prob.iloc[0]
        c1.metric("Favorito", top["PAIS"], f'{top["P_CAMPEON"]:.1f}% campeón')
        c2.metric("Equipos", len(prob))
        c3.metric("Grupos", prob["GRUPO"].nunique())
        c4.metric("Simulaciones", f"{int(n_sim):,}")

        st.bar_chart(prob.head(15).set_index("PAIS")["P_CAMPEON"])
        st.dataframe(prob, use_container_width=True, height=460)
        st.download_button("⬇️ Descargar CSV", prob.to_csv(index=False),
                           "probabilidades_mundial.csv", "text/csv")
    else:
        st.info("Pulsa **Simular torneo** para calcular las probabilidades.")

# ----------------------------------------------------------
# TAB 2: PREDECIR PARTIDO
# ----------------------------------------------------------
with tab2:
    st.subheader("Predicción de un partido")
    equipos = art["equipos_actuales"]["PAIS"].sort_values().tolist()
    c1, c2, c3, c4 = st.columns(4)
    local = c1.selectbox("Local", equipos, index=equipos.index("Mexico") if "Mexico" in equipos else 0)
    visitante = c2.selectbox("Visitante", equipos, index=1)
    neutral = c3.selectbox("Sede neutral", [("No", 0), ("Sí", 1)], format_func=lambda x: x[0])[1]
    torneo = c4.selectbox("Torneo", [("Mundial", 2.0), ("Euro/Copa América", 1.8),
                                     ("Eliminatoria", 1.5), ("Nations League", 1.3),
                                     ("Amistoso", 0.7), ("Otro", 1.0)],
                          format_func=lambda x: x[0])[1]

    if local == visitante:
        st.warning("Elige equipos distintos.")
    else:
        r = mc.predict_match(art, local, visitante, neutral, torneo)
        st.write("**Goles esperados:**",
                 " | ".join(f"{k}: {local} {v[0]} – {v[1]} {visitante}" for k, v in r["goles"].items()))

        filas = []
        for met, p in r["metodos"].items():
            filas.append({"Método": met, f"Gana {local}": round(p[2] * 100, 1),
                          "Empate": round(p[1] * 100, 1), f"Gana {visitante}": round(p[0] * 100, 1)})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        st.caption("Marcadores más probables (Dixon-Coles)")
        st.dataframe(r["marcadores"], use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TAB 3: RANKING ELO
# ----------------------------------------------------------
with tab3:
    st.subheader("Ranking ELO (historia completa)")
    st.dataframe(art["ranking"], use_container_width=True, height=600)

# ----------------------------------------------------------
# TAB 4: EVALUACIÓN
# ----------------------------------------------------------
with tab4:
    st.subheader("Comparativa de métodos en TEST")
    st.caption("LogLoss y Brier: menor es mejor · Accuracy: mayor es mejor")
    st.dataframe(art["tabla_eval"], use_container_width=True, hide_index=True)
