# =============================================================
#  IPL Score Predictor  -  Professional Streamlit Dashboard
#  Model : ANN (ipl_ann_model.keras)  |  Task : predict final score
#  Keep these files in the same folder as app.py :
#       ipl_ann_model.keras , scaler.pkl , encoders.pkl
# =============================================================
import os
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 1. PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="IPL Score Predictor | ANN",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="auto",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Short names and colours for every team (used in the scoreboard)
TEAM_INFO = {
    "Chennai Super Kings": ("CSK", "#F9CD05"),
    "Deccan Chargers": ("DEC", "#4F9BD9"),
    "Delhi Daredevils": ("DD", "#3E7BD6"),
    "Gujarat Lions": ("GL", "#E8602C"),
    "Kings XI Punjab": ("KXIP", "#DD1F2D"),
    "Kochi Tuskers Kerala": ("KTK", "#9B59D0"),
    "Kolkata Knight Riders": ("KKR", "#8E63D8"),
    "Mumbai Indians": ("MI", "#2D7DD2"),
    "Pune Warriors": ("PWI", "#2F9BE3"),
    "Rajasthan Royals": ("RR", "#EA1A85"),
    "Rising Pune Supergiant": ("RPS", "#D11D9B"),
    "Rising Pune Supergiants": ("RPS", "#D11D9B"),
    "Royal Challengers Bangalore": ("RCB", "#EC1C24"),
    "Sunrisers Hyderabad": ("SRH", "#FF822A"),
}


def team_short(name):
    return TEAM_INFO.get(name, (name[:3].upper(), "#FF9F1C"))[0]


def team_color(name):
    return TEAM_INFO.get(name, (name[:3].upper(), "#FF9F1C"))[1]


# ------------------------------------------------------------
# ------------------------------------------------------------
# 2. CUSTOM CSS  (Responsive across Mobile, Tablet & Desktop)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* clean up default Streamlit items */
    #MainMenu, footer, [data-testid="stToolbar"] {display: none;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        visibility: visible !important;
    }

    /* use the full width of the screen */
    .block-container {
        padding: 1.2rem 2.5rem 2rem 2.5rem;
        max-width: 100%;
    }

    /* ---------- sidebar base styling ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111A3A 0%, #0B132B 100%);
        border-right: 1px solid #26335F;
    }

    /* LOCK SIDEBAR ONLY ON DESKTOP (> 768px) */
    @media (min-width: 769px) {
        section[data-testid="stSidebar"] {
            transform: translateX(0px) !important;
            visibility: visible !important;
            min-width: 310px !important;
            max-width: 310px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: 0px !important;
            transform: translateX(0px) !important;
        }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        section[data-testid="stSidebar"] button[kind="header"],
        section[data-testid="stSidebar"] button[kind="headerNoPadding"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }
    }

    .side-title {
        font-size: 1.5rem; font-weight: 800; color: #FF9F1C; margin-bottom: 0;
    }
    .side-sub {color: #8FA0C4; font-size: .85rem; margin-bottom: 1rem;}
    .side-head {
        font-size: .8rem; font-weight: 700; letter-spacing: 1.5px;
        color: #FF9F1C; margin: 1rem 0 .2rem 0; text-transform: uppercase;
    }

    /* ---------- hero ---------- */
    .hero {
        background: linear-gradient(110deg, #FF9F1C 0%, #E71D36 50%, #5F0A87 100%);
        padding: 1.6rem 2rem; border-radius: 20px;
        box-shadow: 0 10px 30px rgba(231, 29, 54, .30);
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 1rem; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: white; font-size: 2.1rem; font-weight: 800;}
    .hero p {margin: .3rem 0 0 0; color: #FFF3E0; font-size: 1rem;}
    .pill {
        display: inline-block; background: rgba(255,255,255,.18);
        color: white; padding: .35rem .9rem; border-radius: 999px;
        font-size: .85rem; font-weight: 600; margin: .2rem .4rem .2rem 0;
        border: 1px solid rgba(255,255,255,.3);
    }

    /* ---------- scoreboard ---------- */
    .scoreboard {
        background: linear-gradient(135deg, #1C2541 0%, #131B35 100%);
        border: 1px solid #2E3D6B; border-radius: 20px;
        padding: 1.4rem 2rem; box-shadow: 0 8px 24px rgba(0,0,0,.35);
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 1rem;
    }
    .sb-team {text-align: center; min-width: 150px;}
    .sb-badge {
        display: inline-block; font-weight: 900; font-size: 1.6rem;
        padding: .5rem 1.1rem; border-radius: 14px; color: #0B132B;
    }
    .sb-name {color: #A9B8D0; font-size: .85rem; margin-top: .4rem;}
    .sb-score {text-align: center;}
    .sb-score .big {font-size: 3.2rem; font-weight: 900; color: white; line-height: 1;}
    .sb-score .ov {color: #FF9F1C; font-weight: 700; font-size: 1.05rem;}
    .sb-score .venue {color: #8FA0C4; font-size: .85rem; margin-top: .3rem;}
    .sb-vs {font-size: 1.2rem; font-weight: 800; color: #5C6F9E;}

    /* ---------- small stat cards ---------- */
    .stat {
        background: #1C2541; border: 1px solid #2E3D6B;
        border-top: 4px solid #FF9F1C; border-radius: 14px;
        padding: .9rem 1rem; text-align: center;
        margin-bottom: 0.6rem;
    }
    .stat .v {font-size: 1.7rem; font-weight: 800; color: white;}
    .stat .l {font-size: .8rem; color: #A9B8D0; letter-spacing: .5px;}

    /* ---------- special effects animations ---------- */
    @keyframes popInGlow {
        0% {
            transform: scale(0.92);
            opacity: 0;
            box-shadow: 0 0 0px rgba(0, 176, 155, 0);
        }
        70% {
            transform: scale(1.03);
            box-shadow: 0 0 35px rgba(150, 201, 61, 0.75);
        }
        100% {
            transform: scale(1);
            opacity: 1;
            box-shadow: 0 10px 30px rgba(0, 176, 155, 0.35);
        }
    }

    @keyframes slideInRight {
        0% {
            transform: translateX(25px);
            opacity: 0;
        }
        100% {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* ---------- prediction card ---------- */
    .pred {
        background: linear-gradient(135deg, #00B09B 0%, #96C93D 100%);
        border-radius: 22px; padding: 1.8rem; text-align: center;
        box-shadow: 0 10px 30px rgba(0,176,155,.35); height: 100%;
        animation: popInGlow 0.65s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    .pred .label {color: #F0FFF4; letter-spacing: 3px; font-size: .95rem; font-weight: 600;}
    .pred .score {color: white; font-size: 5rem; font-weight: 900; line-height: 1.05;}
    .pred .sub {color: #F0FFF4; font-size: 1rem;}

    .compare {
        background: #1C2541; border: 1px solid #2E3D6B; border-radius: 20px;
        padding: 1.4rem 1.6rem; height: 100%;
        animation: slideInRight 0.6s ease-out forwards;
    }
    .compare h4 {color: #FF9F1C; margin: 0 0 .8rem 0; letter-spacing: 1px;}
    .row {display: flex; justify-content: space-between; padding: .45rem 0;
          border-bottom: 1px solid #26335F; color: #D5DDEC;}
    .row b {color: white;}

    .section {font-size: 1.1rem; font-weight: 700; color: #FF9F1C; margin: 1.3rem 0 .5rem 0;}

    /* ---------- button ---------- */
    div.stButton > button {
        width: 100%; height: 3.1rem; font-size: 1.1rem; font-weight: 800;
        border: none; border-radius: 12px; color: white;
        background: linear-gradient(90deg, #FF9F1C, #E71D36);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); color: white;
        box-shadow: 0 8px 20px rgba(231, 29, 54, .4);
    }
    div.stButton > button:disabled {opacity: .4;}

    /* tabs */
    button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}

    /* ============================================================
       MEDIA QUERIES FOR TABLETS & MOBILE DEVICES
       ============================================================ */

    /* Tablets & Small Laptops (max-width: 1024px) */
    @media (max-width: 1024px) {
        .block-container {
            padding: 1rem 1.5rem 2rem 1.5rem;
        }
        .hero h1 {font-size: 1.75rem;}
        .sb-score .big {font-size: 2.6rem;}
        .pred .score {font-size: 4rem;}
    }

    /* Mobile Phones (max-width: 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding: 3.2rem 1rem 1.5rem 1rem !important;
        }

        /* Make the mobile sidebar toggle arrow clearly visible */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            background-color: #FF9F1C !important;
            color: #0B132B !important;
            border-radius: 8px !important;
            top: 0.6rem !important;
            left: 0.6rem !important;
        }

        /* Hero compact layout */
        .hero {
            padding: 1.2rem;
            border-radius: 16px;
        }
        .hero h1 {font-size: 1.45rem;}
        .hero p {font-size: 0.88rem;}
        .pill {
            font-size: 0.72rem;
            padding: 0.25rem 0.65rem;
        }

        /* Scoreboard mobile stacking */
        .scoreboard {
            flex-direction: column;
            padding: 1.2rem 1rem;
            text-align: center;
            gap: 0.8rem;
        }
        .sb-team {
            min-width: unset;
            width: 100%;
        }
        .sb-badge {
            font-size: 1.3rem;
            padding: 0.4rem 0.9rem;
        }
        .sb-score .big {font-size: 2.5rem;}
        .sb-score .ov {font-size: 0.95rem;}
        .sb-vs {font-size: 1rem;}

        /* Stat cards */
        .stat {padding: 0.75rem;}
        .stat .v {font-size: 1.4rem;}
        .stat .l {font-size: 0.72rem;}

        /* Prediction & Outlook cards */
        .pred {
            padding: 1.4rem 1rem;
            margin-bottom: 1rem;
        }
        .pred .label {font-size: 0.8rem; letter-spacing: 2px;}
        .pred .score {font-size: 3.6rem;}
        .pred .sub {font-size: 0.88rem;}

        .compare {padding: 1.1rem;}
        .row {font-size: 0.9rem;}

        /* Tabs fit better on narrow screens */
        button[data-baseweb="tab"] {
            font-size: 0.85rem;
            padding: 0.4rem 0.6rem;
        }
    }

    /* Extra Small Phones (max-width: 400px) */
    @media (max-width: 400px) {
        .hero h1 {font-size: 1.25rem;}
        .sb-score .big {font-size: 2.1rem;}
        .pred .score {font-size: 3rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 3. LOAD MODEL, SCALER, ENCODERS  (cached = loads only once)
# ------------------------------------------------------------
@st.cache_resource
def load_files():
    from tensorflow import keras  # imported here so the page opens fast

    model = keras.models.load_model(
        os.path.join(BASE_DIR, "ipl_ann_model.keras"), compile=False
    )
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


try:
    model, scaler, encoders = load_files()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

teams = list(encoders["bat_team"].classes_)
venues = list(encoders["venue"].classes_)


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------
def overs_to_balls(overs):
    """10.3 overs (cricket style) -> 63 balls"""
    full = int(overs)
    balls = int(round((overs - full) * 10))
    return full * 6 + balls, balls


def predict_many(rows):
    """
    rows = list of tuples in TRAINING ORDER:
    (bat_team, bowl_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5)
    Returns a numpy array with one predicted score for each row.
    """
    data = []
    for bat, bowl, ven, runs, wk, ov, r5, w5 in rows:
        data.append([
            encoders["bat_team"].transform([bat])[0],
            encoders["bowl_team"].transform([bowl])[0],
            encoders["venue"].transform([ven])[0],
            runs, wk, ov, r5, w5,
        ])
    scaled = scaler.transform(np.array(data, dtype=float))
    return model.predict(scaled, verbose=0).flatten()


# ------------------------------------------------------------
# 5. SIDEBAR  (all the inputs live here)
# ------------------------------------------------------------
with st.sidebar:
    st.sidebar.markdown(
        """
        ⭐ **Like this project?** [![GitHub stars](https://img.shields.io/github/stars/chandru-pro-svg/IPL-SCORE-PREDICTION-ANN-MODEL?style=social)](https://github.com/chandru-pro-svg/IPL-SCORE-PREDICTION-ANN-MODEL)
        """
    )
    st.markdown('<div class="side-title">🏏 IPL Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Set the match situation below</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-head">Match details</div>', unsafe_allow_html=True)
    bat_team = st.selectbox("Batting Team", teams, index=0)
    bowl_team = st.selectbox("Bowling Team", [t for t in teams if t != bat_team], index=0)
    default_venue = venues.index("Wankhede Stadium") if "Wankhede Stadium" in venues else 0
    venue = st.selectbox("Venue", venues, index=default_venue)

    st.markdown('<div class="side-head">Current situation</div>', unsafe_allow_html=True)
    runs = st.number_input("Runs Scored", 0, 300, 85, 1)
    overs = st.number_input("Overs Completed (10.3 = 10 overs 3 balls)", 5.0, 19.5, 9.0, 0.1, format="%.1f")
    wickets = st.slider("Wickets Fallen", 0, 9, 3)

    st.markdown('<div class="side-head">Last 5 overs</div>', unsafe_allow_html=True)
    runs_last_5 = st.number_input("Runs in Last 5 Overs", 0, 120, 40, 1)
    wickets_last_5 = st.slider("Wickets in Last 5 Overs", 0, 5, 2)

    # ----- check the inputs before allowing a prediction -----
    balls_done, extra_balls = overs_to_balls(overs)
    errors = []
    if extra_balls > 5:
        errors.append("Overs: after 10.5 comes 11.0 (use .0 to .5 only).")
    if runs_last_5 > runs:
        errors.append("Last-5-overs runs cannot be more than total runs.")
    if wickets_last_5 > wickets:
        errors.append("Last-5-overs wickets cannot be more than total wickets.")
    for msg in errors:
        st.error(msg)

    st.write("")
    predict_clicked = st.button("🚀 Predict Final Score", disabled=len(errors) > 0)

# One tuple that describes the current inputs (used to detect changes)
inputs = (bat_team, bowl_team, venue, runs, wickets, overs, runs_last_5, wickets_last_5)

# ------------------------------------------------------------
# 6. RUN THE PREDICTION WHEN THE BUTTON IS CLICKED
# ------------------------------------------------------------
if predict_clicked and not errors:
    with st.spinner("🏏 The ANN is calculating the final projection..."):
        time.sleep(0.5)
        score = float(predict_many([inputs])[0])
        final_predicted_score = max(int(round(score)), runs)
    
    st.session_state["result"] = {"inputs": inputs, "score": final_predicted_score}
    
    # 1. Trigger floating balloons across the screen
    st.balloons()
    
    # 2. Trigger pop-up toast alert in the top-right corner
    st.toast(f"🏏 Prediction Ready! Projected Score: {final_predicted_score}", icon="🔥")
    
    # 3. Extra snow celebration if it's a massive 200+ score
    if final_predicted_score >= 200:
        st.snow()

result = st.session_state.get("result")
result_is_fresh = result is not None and result["inputs"] == inputs

# Numbers we show on the page
balls_left = 120 - balls_done
crr = runs / (balls_done / 6) if balls_done > 0 else 0
last5_rr = runs_last_5 / 5
rr_projection = int(round(runs + crr * balls_left / 6))

# ------------------------------------------------------------
# 7. MAIN PAGE
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div>
            <h1>🏏 IPL Score Predictor</h1>
            <p>Deep-learning powered final innings score forecast</p>
        </div>
        <div>
            <span class="pill">🧠 ANN Model</span>
            <span class="pill">📊 8 Input Features</span>
            <span class="pill">⚡ Real-time</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 What-If Analysis", "ℹ️ About Model"])

# ===================== TAB 1 : PREDICTION =====================
with tab1:
    # ----- live scoreboard -----
    st.markdown(
        f"""
        <div class="scoreboard">
            <div class="sb-team">
                <span class="sb-badge" style="background:{team_color(bat_team)}">{team_short(bat_team)}</span>
                <div class="sb-name">{bat_team} (batting)</div>
            </div>
            <div class="sb-score">
                <div class="big">{runs}/{wickets}</div>
                <div class="ov">{overs:.1f} overs</div>
                <div class="venue">📍 {venue}</div>
            </div>
            <div class="sb-vs">VS</div>
            <div class="sb-team">
                <span class="sb-badge" style="background:{team_color(bowl_team)}">{team_short(bowl_team)}</span>
                <div class="sb-name">{bowl_team} (bowling)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.markdown(f'<div class="stat"><div class="v">{crr:.2f}</div><div class="l">CURRENT RUN RATE</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat"><div class="v">{last5_rr:.2f}</div><div class="l">LAST 5 OVERS RR</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="stat"><div class="v">{balls_left}</div><div class="l">BALLS REMAINING</div></div>', unsafe_allow_html=True)
    s4.markdown(f'<div class="stat"><div class="v">{10 - wickets}</div><div class="l">WICKETS IN HAND</div></div>', unsafe_allow_html=True)
    s5.markdown(f'<div class="stat"><div class="v">{rr_projection}</div><div class="l">RUN-RATE PROJECTION</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">🔮 Prediction Result</div>', unsafe_allow_html=True)

    if not result_is_fresh:
        if result is None:
            st.info("👈 Set the match details in the sidebar and click **Predict Final Score**.")
        else:
            st.warning("Inputs changed. Click **Predict Final Score** again to refresh the result.")
    else:
        predicted = result["score"]
        extra_runs = predicted - runs
        final_rr = predicted / 20
        difference = predicted - rr_projection

        col_a, col_b = st.columns([1, 1], gap="large")
        with col_a:
            st.markdown(
                f"""
                <div class="pred">
                    <div class="label">PREDICTED FINAL SCORE</div>
                    <div class="score">{predicted}</div>
                    <div class="sub">{team_short(bat_team)} vs {team_short(bowl_team)} &nbsp;•&nbsp; 20 overs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""
                <div class="compare">
                    <h4>MATCH OUTLOOK</h4>
                    <div class="row"><span>Runs still to come</span><b>{extra_runs}</b></div>
                    <div class="row"><span>Expected final run rate</span><b>{final_rr:.2f}</b></div>
                    <div class="row"><span>ANN prediction</span><b>{predicted}</b></div>
                    <div class="row"><span>Simple run-rate projection</span><b>{rr_projection}</b></div>
                    <div class="row"><span>ANN vs run-rate</span><b>{difference:+d} runs</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.progress(min(predicted / 250, 1.0), text=f"Predicted score on a 0 - 250 scale: {predicted}")

# ===================== TAB 2 : WHAT-IF ANALYSIS =====================
with tab2:
    if not result_is_fresh:
        st.info("Make a prediction first, then come here to see how the score changes.")
    else:
        st.caption("These charts ask the same ANN model 'what if?' questions by changing one input at a time.")

        # ----- Chart 1 : change runs in last 5 overs -----
        r5_values = list(range(0, min(runs, 100) + 1, 5))
        rows = [(bat_team, bowl_team, venue, runs, wickets, overs, r, wickets_last_5) for r in r5_values]
        df1 = pd.DataFrame({"Predicted score": predict_many(rows).round(0)}, index=r5_values)
        df1.index.name = "Runs in last 5 overs"

        # ----- Chart 2 : change wickets fallen -----
        wk_values = list(range(0, 10))
        rows = [(bat_team, bowl_team, venue, runs, w, overs, runs_last_5, min(wickets_last_5, w)) for w in wk_values]
        df2 = pd.DataFrame({"Predicted score": predict_many(rows).round(0)}, index=wk_values)
        df2.index.name = "Wickets fallen"

        # ----- Chart 3 : change the venue -----
        rows = [(bat_team, bowl_team, v, runs, wickets, overs, runs_last_5, wickets_last_5) for v in venues]
        df3 = pd.DataFrame({"Venue": venues, "Predicted score": predict_many(rows).round(0)})
        df3 = df3.sort_values("Predicted score", ascending=False).head(10).set_index("Venue")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section">📈 Effect of last-5-overs momentum</div>', unsafe_allow_html=True)
            st.line_chart(df1, color="#FF9F1C")
        with c2:
            st.markdown('<div class="section">🎳 Effect of wickets fallen</div>', unsafe_allow_html=True)
            st.bar_chart(df2, color="#E71D36")

        st.markdown('<div class="section">🏟️ Top 10 venues for this match-up</div>', unsafe_allow_html=True)
        st.bar_chart(df3, color="#00B09B", horizontal=True)

# ===================== TAB 3 : ABOUT MODEL =====================
with tab3:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown('<div class="section">🧠 Model architecture</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Layer": ["Input", "Dense + ReLU", "Dropout (0.2)", "Dense + ReLU", "Dropout (0.2)", "Dense + ReLU", "Output (linear)"],
            "Size": ["8", "128", "-", "64", "-", "32", "1"],
        }).set_index("Layer"))
        st.caption("Type: regression ANN  •  Loss: MSE  •  Metric: MAE  •  Optimizer: Adam")
    with a2:
        st.markdown('<div class="section">🧾 Input features</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Feature": ["bat_team", "bowl_team", "venue", "runs", "wickets", "overs", "runs_last_5", "wickets_last_5"],
            "Processing": ["Label encoded + scaled"] * 3 + ["Scaled"] * 5,
        }).set_index("Feature"))
    st.info("Predictions are estimates from historical IPL data. Real matches can change quickly!")

# ------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------
st.markdown(
    "<p style='text-align:center; color:#6B7BA0; margin-top:2rem;'>Built with Streamlit & TensorFlow • IPL ANN Score Predictor</p>",
    unsafe_allow_html=True,
)
