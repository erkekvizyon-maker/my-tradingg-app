import streamlit as st
import pandas as pd
import os
from datetime import datetime
from catboost import CatBoostClassifier

DF_PATH = "senaryo13.csv"
OUT_PATH = "giris_kayit_arsiv.csv"

@st.cache_data
def load_model_and_data():
    df = pd.read_csv(DF_PATH)
    df["y"] = df["label"].map({"ayni": 1, "ters": 0})
    df["slope_sol"] = df["sol_y"] / (df["sol_x"] + 1e-9)
    df["slope_sag"] = df["sag_y"] / (df["sag_x"] + 1e-9)
    df["int_sol_calc"] = df["sol_poc"] / (df["sol_total"] + 1e-9)
    df["int_sag_calc"] = df["sag_poc"] / (df["sag_total"] + 1e-9)
    df["strength_sol"] = df["slope_sol"] * df["int_sol_calc"]
    df["strength_sag"] = df["slope_sag"] * df["int_sag_calc"]
    df["diff_strength"] = (df["strength_sag"] - df["strength_sol"]) / (
        df["strength_sag"] + df["strength_sol"] + 1e-9)
    df["pivot_enc"] = df["pivot"].map({"tepe": 1, "dip": 0})
    df["tip_enc"] = df["tip"]
    df["poc_dist_enc"] = df["poc_dist"].map({"yakin": 1, "uzak": 0})

    feature_cols = [
        "pivot_enc", "tip_enc", "poc_dist_enc",
        "slope_sol", "slope_sag", "int_sol_calc", "int_sag_calc", "diff_strength"
    ]
    cat_features = [0, 1, 2]
    Xc = df[feature_cols]
    y = df["y"].values
    model = CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.05,
        loss_function="Logloss", verbose=False, random_seed=42)
    model.fit(Xc, y, cat_features=cat_features)
    return model

model = load_model_and_data()

st.set_page_config(layout="wide")
st.title("Trading Çekirdek Algoritma Giriş ve Backtest")

if "selected_row" not in st.session_state:
    st.session_state.selected_row = None

default_vals = {
    "pivot": "tepe", "tip": 1,
    "sol_y": 1.0, "sol_x": 1.0, "sol_poc": 1.0, "sol_total": 1.0,
    "sag_y": 1.0, "sag_x": 1.0, "sag_poc": 1.0, "sag_total": 1.0,
    "poc_dist": "yakin", "gercek_yon": "ayni"
}
if st.session_state.selected_row is not None:
    for k in default_vals:
        if k in st.session_state.selected_row:
            default_vals[k] = st.session_state.selected_row[k]
    st.session_state.selected_row = None

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("📋 Veri Girişi")
    pivot = st.selectbox("1. Pivot tipi", options=["tepe", "dip"], index=0 if default_vals["pivot"] == "tepe" else 1)
    tip = st.selectbox("2. Tip", options=[1, 2], index=default_vals["tip"]-1)

    sol_y = st.number_input("3. Sol Kol Y", value=default_vals["sol_y"])
    sol_x = st.number_input("4. Sol Kol X", value=default_vals["sol_x"])
    slope_sol = sol_y / (sol_x + 1e-9)
    st.caption(f"5. Sol Slope (Y/X): {slope_sol:.3f}")

    sol_poc = st.number_input("6. Sol Kol POC hacim", value=default_vals["sol_poc"])
    sol_total = st.number_input("7. Sol Kol Toplam hacim", value=default_vals["sol_total"])
    int_sol_calc = sol_poc / (sol_total + 1e-9)
    st.caption(f"8. Sol Intensity (POC/Total): {int_sol_calc:.3f}")

    sag_y = st.number_input("9. Sağ Kol Y", value=default_vals["sag_y"])
    sag_x = st.number_input("10. Sağ Kol X", value=default_vals["sag_x"])
    slope_sag = sag_y / (sag_x + 1e-9)
    st.caption(f"11. Sağ Slope (Y/X): {slope_sag:.3f}")

    sag_poc = st.number_input("12. Sağ Kol POC hacim", value=default_vals["sag_poc"])
    sag_total = st.number_input("13. Sağ Kol Toplam hacim", value=default_vals["sag_total"])
    int_sag_calc = sag_poc / (sag_total + 1e-9)
    st.caption(f"14. Sağ Intensity (POC/Total): {int_sag_calc:.3f}")

    big_intensity_col = "sol" if int_sol_calc > int_sag_calc else "sag"
    st.caption(f"15. İntensity Büyük Kol: {big_intensity_col.upper()}")

    sol_strength = slope_sol * int_sol_calc
    sag_strength = slope_sag * int_sag_calc
    st.caption(f"16. Sol Strength: {sol_strength:.3f}")
    st.caption(f"16. Sağ Strength: {sag_strength:.3f}")
    strong_col = "SOL" if sol_strength > sag_strength else "SAĞ"
    st.caption(f"16. Güçlü Kol: {strong_col}")

    diff_strength = (sag_strength - sol_strength) / (sag_strength + sol_strength + 1e-9)
    st.caption(f"17. Strength Farkı: {diff_strength:.3f}")

    poc_dist = st.selectbox("18. POC mesafesi (pivota yakın mı?)", options=["yakin", "uzak"], index=0 if default_vals["poc_dist"]=="yakin" else 1)

    if st.button("19. Hesapla ve Kaydet"):
        feature_vec = pd.DataFrame([{
            "pivot_enc":    1 if pivot == 'tepe' else 0,
            "tip_enc":      int(tip),
            "poc_dist_enc": 1 if poc_dist == 'yakin' else 0,
            "slope_sol":    slope_sol,
            "slope_sag":    slope_sag,
            "int_sol_calc": int_sol_calc,
            "int_sag_calc": int_sag_calc,
            "diff_strength": diff_strength
        }])
        prob = model.predict_proba(feature_vec)[0, 1]
        pred = 'ayni' if prob >= 0.5 else 'ters'
        st.success(f"MODEL TAHMİNİ: **{pred.upper()}** (Prob: {prob:.3f})")
        st.info(f"Güçlü Kol: {strong_col} | İnt. Büyük Kol: {big_intensity_col.upper()}")
        gercek_yon = st.radio("20. Gerçek yön (modelle aynıysa GEÇERLİ, farklıysa GEÇERSİZ):", options=["ayni", "ters"], horizontal=True,
                              index=0 if default_vals["gercek_yon"]=="ayni" else 1)
        gecerlilik = "GEÇERLİ" if (pred == gercek_yon) else "GEÇERSİZ"
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pivot, tip, sol_x, sol_y, sag_x, sag_y, sol_total, sag_total, sol_poc, sag_poc,
            "", "", "", "", "", "",
            big_intensity_col, poc_dist, gercek_yon, pred, prob, gecerlilik,
            slope_sol, int_sol_calc, sol_strength, slope_sag, int_sag_calc, sag_strength, strong_col, diff_strength
        ]
        header = [
            "timestamp", "pivot", "tip", "sol_x", "sol_y", "sag_x", "sag_y",
            "sol_total", "sag_total", "sol_poc", "sag_poc",
            "sol_poc_dist", "sag_poc_dist", "nx_sol", "nx_sag", "ny_sol", "ny_sag",
            "big_poc_col", "poc_dist", "gercek_label", "model_pred", "model_prob", "gecerlilik",
            "slope_sol", "int_sol_calc", "sol_strength", "slope_sag", "int_sag_calc", "sag_strength", "strong_col", "diff_strength"
        ]
        file_exists = os.path.isfile(OUT_PATH)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(",".join(header) + "\n")
            f.write(",".join(map(str, row)) + "\n")
        st.success(f"Satır kaydedildi! ({OUT_PATH}) -- {gecerlilik}")

with col2:
    st.header("🗒️ Kayıt Tablosu")
    if os.path.isfile(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        # SADECE DOĞRU GERÇERLİ/GEÇERSİZ TANIMI:
        # GEÇERLİ: model_pred == gercek_label, GEÇERSİZ: değil
        df["gecerlilik"] = (df["model_pred"].astype(str) == df["gercek_label"].astype(str)).map({True: "GEÇERLİ", False: "GEÇERSİZ"})
        st.dataframe(df, use_container_width=True)

        st.markdown("Satır seç ve işlemini belirt:")
        row_index = st.number_input("Satır (index) seç", min_value=0, max_value=len(df)-1, value=0, step=1)
        if st.button("Doldur (Formu Otomatik Yükle)"):
            row = df.iloc[row_index]
            st.session_state.selected_row = {
                "pivot": row["pivot"], "tip": int(row["tip"]),
                "sol_x": float(row["sol_x"]), "sol_y": float(row["sol_y"]),
                "sol_poc": float(row["sol_poc"]), "sol_total": float(row["sol_total"]),
                "sag_x": float(row["sag_x"]), "sag_y": float(row["sag_y"]),
                "sag_poc": float(row["sag_poc"]), "sag_total": float(row["sag_total"]),
                "poc_dist": row["poc_dist"], "gercek_yon": row["gercek_label"],
            }
            st.experimental_rerun()
        if st.button("Sil (Seçili Satır)"):
            df.drop(index=row_index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(OUT_PATH, index=False)
            st.experimental_rerun()
    else:
        st.info("Henüz veri kaydedilmedi.")