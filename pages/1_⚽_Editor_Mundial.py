"""
pages/1_⚽_Editor_Mundial.py — Editor gráfico de resultados del Mundial 2026
============================================================================
Página independiente (aparece en el drawer/barra lateral de Streamlit) para
editar IN SITU solo los partidos del Mundial, organizados por fase, con cards
por partido (bandera + nombre + marcador), opción de traer de ESPN y commit
directo al CSV en GitHub. No toca el resto del dataset ni la app principal.
"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from mundial import modelo_core as mc
from mundial import ingesta as ing
from mundial import nombres

st.set_page_config(page_title="Editor Mundial 2026", page_icon="⚽", layout="wide")

# ============================================================
# Banderas emoji (país -> ISO2; Inglaterra/Escocia/Gales especiales)
# ============================================================
ISO2 = {
    "Algeria": "DZ", "Argentina": "AR", "Australia": "AU", "Austria": "AT",
    "Belgium": "BE", "Bosnia and Herzegovina": "BA", "Brazil": "BR", "Canada": "CA",
    "Cape Verde": "CV", "Colombia": "CO", "Croatia": "HR", "Curaçao": "CW",
    "Czech Republic": "CZ", "DR Congo": "CD", "Ecuador": "EC", "Egypt": "EG",
    "France": "FR", "Germany": "DE", "Ghana": "GH", "Haiti": "HT", "Iran": "IR",
    "Iraq": "IQ", "Ivory Coast": "CI", "Japan": "JP", "Jordan": "JO", "Mexico": "MX",
    "Morocco": "MA", "Netherlands": "NL", "New Zealand": "NZ", "Norway": "NO",
    "Panama": "PA", "Paraguay": "PY", "Portugal": "PT", "Qatar": "QA",
    "Saudi Arabia": "SA", "Senegal": "SN", "South Africa": "ZA", "South Korea": "KR",
    "Spain": "ES", "Sweden": "SE", "Switzerland": "CH", "Tunisia": "TN",
    "Turkey": "TR", "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
    "Colombia ": "CO",
}


def _subdiv(code):
    base = "\U0001F3F4"
    return base + "".join(chr(0xE0000 + ord(c)) for c in code) + "\U000E007F"


ESPECIALES = {"England": _subdiv("gbeng"), "Scotland": _subdiv("gbsct"),
              "Wales": _subdiv("gbwls")}


def bandera(team):
    if team in ESPECIALES:
        return ESPECIALES[team]
    code = ISO2.get(team)
    if not code:
        return "🏳️"
    return chr(0x1F1E6 + ord(code[0]) - 65) + chr(0x1F1E6 + ord(code[1]) - 65)


# ============================================================
# Datos del Mundial (sin entrenar el modelo)
# ============================================================
CSV = mc.CSV_DEFAULT


def _norm_df(df):
    d = df.copy()
    for c in ("home_team", "away_team"):
        d[c] = d[c].map(nombres.normaliza)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d


@st.cache_data(show_spinner=False)
def cargar(_sig):
    df_full, por_jugar = mc.load_data()
    groups, _, _ = mc.reconstruir_grupos({"df": df_full, "por_jugar": por_jugar})
    wc_j = df_full[(df_full["tournament"] == "FIFA World Cup")
                   & (df_full["date"] >= "2026-06-01")]
    wc = pd.concat([wc_j, por_jugar[por_jugar["tournament"] == "FIFA World Cup"]],
                   ignore_index=True)
    wc = _norm_df(wc)
    # dedup por partido (tras normalizar nombres pueden colisionar grafías,
    # p.ej. "Congo DR"/"DR Congo"); conserva la fila que ya trae marcador.
    wc["_con"] = wc[["home_score", "away_score"]].notna().any(axis=1)
    wc = (wc.sort_values("_con")
          .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
          .drop(columns="_con").reset_index(drop=True))
    return wc, groups


sig = dt.date.today().isoformat()
wc, groups = cargar(sig)
team2group = {t: g for g, ts in groups.items() for t in ts}


def clave(date, home, away):
    return f"{date}|{home}|{away}"


def _fase(row):
    gh, ga = team2group.get(row["home_team"]), team2group.get(row["away_team"])
    if gh and gh == ga:
        return f"Grupo {gh}"
    return "Eliminatorias"


wc["fase"] = wc.apply(_fase, axis=1)
# inicializa el estado de cada marcador (una vez) desde el CSV
for _, r in wc.iterrows():
    k = clave(r["date"], r["home_team"], r["away_team"])
    for suf, col in (("h", "home_score"), ("a", "away_score")):
        kk = f"{k}_{suf}"
        if kk not in st.session_state:
            v = r[col]
            st.session_state[kk] = None if pd.isna(v) else int(v)


# ============================================================
# UI
# ============================================================
st.title("⚽ Editor de resultados — Mundial 2026")
st.caption("Edita los marcadores del Mundial por fase, trae resultados de ESPN "
           "y haz commit directo al CSV. Solo afecta a los partidos del Mundial.")

jugados = int(wc[["home_score", "away_score"]].notna().all(axis=1).sum())
c = st.columns(3)
c[0].metric("Partidos del Mundial", len(wc))
c[1].metric("Con marcador", jugados)
c[2].metric("Pendientes", len(wc) - jugados)

# ---- Traer de ESPN ----
with st.expander("🌐 Traer marcadores de ESPN", expanded=False):
    cc = st.columns([1, 1, 2])
    fecha = cc[0].date_input("Fecha a traer", value=dt.date(2026, 6, 17))
    if cc[1].button("Traer ese día"):
        try:
            fx = ing.traer_partidos_del_dia(fecha.isoformat())
            n = 0
            idx = {clave(r["date"], r["home_team"], r["away_team"]): r
                   for _, r in fx.iterrows()}
            for kk, r in idx.items():
                if pd.isna(r["home_score"]):
                    continue
                # empareja contra los partidos del Mundial (cualquier orden)
                for _, w in wc.iterrows():
                    same = ({w["home_team"], w["away_team"]} ==
                            {r["home_team"], r["away_team"]} and w["date"] == r["date"])
                    if same:
                        kw = clave(w["date"], w["home_team"], w["away_team"])
                        if w["home_team"] == r["home_team"]:
                            st.session_state[f"{kw}_h"] = int(r["home_score"])
                            st.session_state[f"{kw}_a"] = int(r["away_score"])
                        else:
                            st.session_state[f"{kw}_h"] = int(r["away_score"])
                            st.session_state[f"{kw}_a"] = int(r["home_score"])
                        n += 1
            st.success(f"Rellenados {n} partido(s) desde ESPN. Revisa y commitea.")
        except Exception as e:
            st.warning(f"No se pudo traer ({type(e).__name__}): {e}")


def tarjeta(row):
    k = clave(row["date"], row["home_team"], row["away_team"])
    h, a = row["home_team"], row["away_team"]
    jugado = (st.session_state.get(f"{k}_h") is not None
              and st.session_state.get(f"{k}_a") is not None)
    with st.container(border=True):
        col = st.columns([0.5, 3, 1.4, 0.4, 1.4, 3, 0.5])
        col[0].markdown(f"<div style='font-size:2rem'>{bandera(h)}</div>",
                        unsafe_allow_html=True)
        col[1].markdown(f"**{h}**<br><span style='color:gray;font-size:0.8rem'>"
                        f"{row['date']}{' · ✅' if jugado else ' · ⏳'}</span>",
                        unsafe_allow_html=True)
        col[2].number_input("GL", key=f"{k}_h", min_value=0, step=1,
                            label_visibility="collapsed")
        col[3].markdown("<div style='text-align:center;font-size:1.3rem'>–</div>",
                        unsafe_allow_html=True)
        col[4].number_input("GV", key=f"{k}_a", min_value=0, step=1,
                            label_visibility="collapsed")
        col[5].markdown(f"<div style='text-align:right'><b>{a}</b><br>"
                        f"<span style='color:gray;font-size:0.8rem'>vs</span></div>",
                        unsafe_allow_html=True)
        col[6].markdown(f"<div style='font-size:2rem;text-align:right'>{bandera(a)}</div>",
                        unsafe_allow_html=True)


fases = ["Fase de grupos", "Eliminatorias"]
t_grupos, t_elim = st.tabs(fases)

with t_grupos:
    gnames = sorted(groups.keys())
    g_sel = st.radio("Grupo", gnames, horizontal=True, key="grupo_sel")
    sub = wc[(wc["fase"] == f"Grupo {g_sel}")].sort_values("date")
    st.markdown(f"##### Grupo {g_sel} · {' · '.join(groups[g_sel])}")
    if sub.empty:
        st.info("Sin partidos para este grupo.")
    for _, r in sub.iterrows():
        tarjeta(r)

with t_elim:
    sub = wc[wc["fase"] == "Eliminatorias"].sort_values("date")
    if sub.empty:
        st.info("Aún no hay partidos de eliminatorias con selecciones definidas "
                "(se rellenan al cerrar la fase de grupos).")
    else:
        for fecha_d, grp in sub.groupby("date"):
            st.markdown(f"##### {fecha_d}")
            for _, r in grp.iterrows():
                tarjeta(r)


# ============================================================
# Guardar / commit
# ============================================================
st.divider()
st.markdown("### 💾 Guardar cambios")


def _csv_actualizado():
    orig_cols = pd.read_csv(CSV, nrows=0).columns.tolist()
    full = _norm_df(pd.read_csv(CSV))
    edits = {}
    for _, r in wc.iterrows():
        k = clave(r["date"], r["home_team"], r["away_team"])
        vh, va = st.session_state.get(f"{k}_h"), st.session_state.get(f"{k}_a")
        edits[(r["date"], r["home_team"], r["away_team"])] = (
            np.nan if vh is None else float(vh),
            np.nan if va is None else float(va))
    hs = full["home_score"].tolist()
    asc = full["away_score"].tolist()
    keys = list(zip(full["date"], full["home_team"], full["away_team"]))
    for i, key3 in enumerate(keys):
        if key3 in edits:
            hs[i], asc[i] = edits[key3]
    full["home_score"], full["away_score"] = hs, asc
    return full[orig_cols]


col = st.columns(2)
col[0].download_button("⬇️ Descargar CSV actualizado",
                       _csv_actualizado().to_csv(index=False),
                       "results_actualizado_mundial_2026.csv", "text/csv")

try:
    gh = dict(st.secrets.get("github", {}))
except Exception:
    gh = {}
with col[1]:
    if not gh.get("token"):
        st.caption("Configura `[github]` en Secrets para commitear desde aquí.")
    else:
        pwd = st.text_input("Contraseña", type="password", key="ed_pwd")
        if st.button("💾 Commit a GitHub", type="primary"):
            import hmac
            if not hmac.compare_digest(str(pwd), str(gh.get("password", ""))):
                st.error("Contraseña incorrecta.")
            else:
                try:
                    from mundial import gh_commit
                    res = gh_commit.commit_file(
                        gh["token"], gh["repo"],
                        gh.get("path", "results_actualizado_mundial_2026.csv"),
                        _csv_actualizado().to_csv(index=False),
                        "Resultados Mundial vía editor", branch=gh.get("branch", "main"))
                    st.success(f"✅ Commit hecho. Redeploy en ~1-2 min.\n\n"
                               f"{res.get('html_url', '')}")
                except Exception as e:
                    st.error(f"Falló el commit: {type(e).__name__}: {e}")
