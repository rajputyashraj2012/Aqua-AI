# ================================================================
# AquaAI — ONE-FILE MULTI-PAGE STREAMLIT APPLICATION
# Working standalone version
#
# Install:
#   python3 -m pip install streamlit pandas numpy scikit-learn joblib reportlab
#
# Run:
#   streamlit run aquaai.py
# ================================================================

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AquaAI | Smart Water Management",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "aquaai_data"

# Some hosting environments (e.g. read-only containers) don't allow writing
# next to the script. Persisting cached CSV/model files is a nice-to-have,
# not a requirement for the app to run, so failures here must never crash
# the app on import.
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_DIR_WRITABLE = True
except OSError:
    _DATA_DIR_WRITABLE = False

DATA_FILE = DATA_DIR / "household_water_data.csv"
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"
MODEL_FILE = DATA_DIR / "water_demand_model.pkl"

RANDOM_SEED = 42
TARGET = "actual_consumption_lpd"

# Baseline flush volume (litres) used as the "before efficiency" comparison
# point when estimating potential water savings from low-flow toilets.
BASELINE_FLUSH_LITRES = 8
EFFICIENT_FLUSH_LITRES = 4

FEATURES = [
    "household_size",
    "temperature_c",
    "rainfall_mm",
    "water_supply_lpd",
    "previous_consumption_lpd",
    "toilet_flushes",
    "flush_litres",
    "greywater_available_lpd",
    "recycled_water_lpd",
    "leakage_lpd",
    "awareness_score",
    "water_stress_index",
    "month",
    "day_of_week",
]


# ----------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "aqua_navigation" not in st.session_state:
    st.session_state.aqua_navigation = "🏠 Overview"


# ----------------------------------------------------------------
# CSS
# ----------------------------------------------------------------
def inject_css(dark: bool = False) -> None:
    """Inject the app's custom CSS theme (light or dark variant)."""
    if dark:
        bg = "linear-gradient(135deg,#06151e,#0a2533 55%,#031017)"
        text = "#e9fbff"
        section_color = "#d8f7ff"
        glass_bg = "rgba(255,255,255,.075)"
        glass_border = "rgba(255,255,255,.12)"
        dashboard_bg = "rgba(255,255,255,.075)"
        dashboard_text = "#bdebf5"
    else:
        bg = (
            "radial-gradient(circle at 10% 10%,rgba(95,224,239,.28),transparent 28%),"
            "radial-gradient(circle at 90% 15%,rgba(0,119,182,.18),transparent 27%),"
            "linear-gradient(135deg,#eafcff,#f8fdff 45%,#eef9ff)"
        )
        text = "#102a43"
        section_color = "#073b59"
        glass_bg = "rgba(255,255,255,.58)"
        glass_border = "rgba(255,255,255,.72)"
        dashboard_bg = "rgba(255,255,255,.58)"
        dashboard_text = "#416b82"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: Inter, sans-serif;
        }}

        .stApp {{
            background: {bg};
            color: {text};
            min-height: 100vh;
        }}

        .block-container {{
            max-width: 1500px;
            padding: 1.4rem 2rem 4rem;
        }}

        /* HERO */
        .hero {{
            position: relative;
            overflow: hidden;
            padding: 38px 42px;
            margin-bottom: 25px;
            border-radius: 30px;
            color: white;
            background: linear-gradient(
                115deg,
                rgba(3,48,71,.98),
                rgba(0,119,182,.88),
                rgba(18,181,203,.78)
            );
            box-shadow: 0 28px 70px rgba(0,92,135,.25);
        }}

        .hero:after {{
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            right: -100px;
            top: -130px;
            border-radius: 50%;
            background: rgba(255,255,255,.09);
        }}

        .hero::before {{
            content: "💧";
            position: absolute;
            right: 7%;
            bottom: 8%;
            font-size: 92px;
            opacity: .13;
            transform: rotate(7deg);
        }}

        .hero h1 {{
            position: relative;
            z-index: 2;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: clamp(40px,5vw,62px);
            margin: 10px 0 0;
            font-weight: 800;
            letter-spacing: -.05em;
        }}

        .hero p {{
            position: relative;
            z-index: 2;
            max-width: 900px;
            font-size: 17px;
            line-height: 1.6;
            color: rgba(255,255,255,.92);
        }}

        .badge {{
            display: inline-block;
            position: relative;
            z-index: 3;
            padding: 7px 12px;
            margin: 3px 4px 5px 0;
            border-radius: 999px;
            background: rgba(255,255,255,.12);
            border: 1px solid rgba(255,255,255,.22);
            font-size: 12px;
            font-weight: 700;
        }}

        .section-title {{
            margin: 28px 0 13px;
            color: {section_color};
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 25px;
            font-weight: 800;
        }}

        /* METRIC CARDS */
        .metric {{
            position: relative;
            overflow: hidden;
            isolation: isolate;
            padding: 24px;
            border-radius: 25px;
            color: white;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border: 1px solid rgba(255,255,255,.28);
            background: linear-gradient(135deg,#177da3,#42bfd1);
            box-shadow:
                0 18px 35px rgba(4,73,103,.16),
                0 5px 12px rgba(4,73,103,.08),
                inset 0 1px 0 rgba(255,255,255,.22);
            animation: aquaMetricFloat 5.8s ease-in-out infinite;
            transition: transform .28s ease, box-shadow .28s ease;
        }}

        .metric::before {{
            content: "";
            position: absolute;
            width: 105px;
            height: 105px;
            right: -18px;
            bottom: -30px;
            border-radius: 50%;
            background: rgba(255,255,255,.15);
            z-index: -1;
        }}

        .metric::after {{
            content: "";
            position: absolute;
            width: 210px;
            height: 70px;
            left: -45px;
            top: -32px;
            border-radius: 50%;
            background: rgba(255,255,255,.08);
            transform: rotate(-10deg);
            z-index: -1;
        }}

        .metric:hover {{
            transform: translateY(-8px) scale(1.015);
            box-shadow: 0 28px 50px rgba(4,73,103,.23);
        }}

        .metric-0 {{ background: linear-gradient(135deg,#207b9f,#42b6cb); }}
        .metric-1 {{ background: linear-gradient(135deg,#17865e,#4cc08e); animation-delay:-.8s; }}
        .metric-2 {{ background: linear-gradient(135deg,#6847a5,#8e6bd1); animation-delay:-1.6s; }}
        .metric-3 {{ background: linear-gradient(135deg,#bd6c17,#efa947); animation-delay:-2.4s; }}
        .metric-4 {{ background: linear-gradient(135deg,#ad3945,#df626c); animation-delay:-3.2s; }}

        .metric h4 {{
            position: relative;
            z-index: 2;
            margin: 0;
            font-size: 15px;
            opacity: .92;
            font-weight: 700;
        }}

        .metric h2 {{
            position: relative;
            z-index: 2;
            margin: 22px 0 0;
            font-size: clamp(27px, 2.2vw, 38px);
            line-height: 1.05;
            letter-spacing: -.035em;
            font-weight: 800;
            text-shadow: 0 2px 12px rgba(0,0,0,.12);
        }}

        @keyframes aquaMetricFloat {{
            0%,100% {{ transform:translateY(0); }}
            50% {{ transform:translateY(-7px); }}
        }}

        /* BUTTONS */
        .stButton > button,
        .stDownloadButton > button {{
            border: 0;
            border-radius: 15px;
            min-height: 46px;
            font-weight: 700;
            color: white;
            background: linear-gradient(135deg,#075985,#12b5cb);
            box-shadow: 0 9px 22px rgba(7,89,133,.20);
            transition: all .25s ease;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            filter: brightness(1.06);
            box-shadow: 0 14px 28px rgba(7,89,133,.27);
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            min-width: 330px;
            background:
                radial-gradient(circle at 15% 15%,rgba(0,212,255,.12),transparent 28%),
                linear-gradient(180deg,#021923 0%,#043b55 50%,#021923 100%);
        }}

        section[data-testid="stSidebar"] > div {{
            padding: 1.1rem .9rem 1.5rem;
        }}

        section[data-testid="stSidebar"] * {{
            color: #ecfeff;
        }}

        .aqua-brand {{
            text-align: center;
            padding: 10px 0 25px;
        }}

        .aqua-drop {{
            font-size: 45px;
            filter: drop-shadow(0 0 15px rgba(0,212,255,.5));
            animation: aqua-float 3s ease-in-out infinite;
        }}

        .aqua-name {{
            color: white;
            font-size: 29px;
            font-weight: 800;
        }}

        .aqua-tagline {{
            color: #7ddcf0;
            font-size: 9px;
            letter-spacing: 2px;
            font-weight: 700;
            margin-top: 4px;
        }}

        .nav-title {{
            color: #ffffff;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 20px 8px 12px;
        }}

        section[data-testid="stSidebar"] .stButton {{
            margin: 0 0 10px 0;
        }}

        section[data-testid="stSidebar"] .stButton > button {{
            position: relative;
            width: 100%;
            min-height: 60px;
            padding: 0 18px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            border-radius: 17px;
            border: 1px solid rgba(180,240,255,.18);
            color: #e9fbff !important;
            background: linear-gradient(
                135deg,
                rgba(255,255,255,.13),
                rgba(255,255,255,.045)
            );
            box-shadow:
                0 10px 25px rgba(0,0,0,.18),
                inset 0 1px 0 rgba(255,255,255,.18);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            font-size: 16px;
            font-weight: 700;
            text-align: left;
            overflow: hidden;
            transition: transform .25s ease, background .25s ease,
                        border-color .25s ease, box-shadow .25s ease;
        }}

        section[data-testid="stSidebar"] .stButton > button::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 48%;
            background: linear-gradient(180deg,rgba(255,255,255,.13),transparent);
            pointer-events: none;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover {{
            transform: translateY(-4px) scale(1.012) !important;
            background: linear-gradient(
                135deg,
                rgba(50,205,235,.27),
                rgba(255,255,255,.08)
            );
            border-color: rgba(150,235,255,.48);
            box-shadow:
                0 16px 32px rgba(0,0,0,.28),
                0 0 20px rgba(50,210,240,.13);
        }}

        .nav-active .stButton > button {{
            background: linear-gradient(
                135deg,
                rgba(35,194,224,.38),
                rgba(70,110,220,.22)
            ) !important;
            border-color: rgba(165,240,255,.68) !important;
            box-shadow:
                0 15px 34px rgba(0,65,95,.34),
                0 0 24px rgba(40,210,240,.18),
                inset 0 1px 0 rgba(255,255,255,.36) !important;
            transform: translateX(4px);
        }}

        .sidebar-status {{
            margin-top: 18px;
            padding: 15px;
            border-radius: 15px;
            background: rgba(255,255,255,.065);
            border: 1px solid rgba(255,255,255,.09);
            color: #bdebf5;
            font-size: 11px;
            line-height: 1.65;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
        }}

        .sidebar-footer {{
            text-align: center;
            color: #6caaba;
            font-size: 9px;
            margin-top: 25px;
            line-height: 1.6;
        }}

        .dashboard-status {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin: 8px 0 20px;
            padding: 15px 20px;
            border-radius: 17px;
            color: {dashboard_text};
            background: {dashboard_bg};
            border: 1px solid rgba(255,255,255,.8);
            box-shadow: 0 10px 28px rgba(5,72,105,.07);
            backdrop-filter: blur(15px);
            font-size: 14px;
        }}

        .dashboard-status strong {{ color: {section_color}; }}

        @keyframes aqua-float {{
            0%,100% {{ transform:translateY(0); }}
            50% {{ transform:translateY(-5px); }}
        }}

        @media(max-width:700px) {{
            .block-container {{ padding: 1rem; }}
            .hero {{ padding: 27px 23px; border-radius: 23px; }}
            .dashboard-status {{
                flex-direction: column;
                align-items: flex-start;
            }}
            section[data-testid="stSidebar"] {{ min-width: 290px; }}
        }}

        @media(prefers-reduced-motion: reduce) {{
            .metric, .aqua-drop {{ animation:none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    """Render the gradient hero banner used at the top of every page."""
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge">💧 AQUA INTELLIGENCE PLATFORM</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="badge">• Random Forest AI</span>
            <span class="badge">• Live Analytics</span>
            <span class="badge">• Water Sustainability</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """Render a section heading."""
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: list[tuple[str, str]]) -> None:
    """Render a row of KPI cards from a list of (label, value) pairs."""
    if not items:
        return
    cols = st.columns(len(items), gap="medium")
    for index, (col, (label, value)) in enumerate(zip(cols, items)):
        with col:
            st.markdown(
                f"""
                <div class="metric metric-{index % 5}">
                    <h4>{label}</h4>
                    <h2>{value}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )


def dashboard_status(record_count: int, r2: float, high_risk_count: int) -> None:
    """Render the small status strip showing record count, model R² and risk."""
    st.markdown(
        f"""
        <div class="dashboard-status">
            <div>
                ◉ <strong>Dashboard ready</strong>
                <span> · {record_count:,} records in view</span>
            </div>
            <div>
                AI model · <strong>Random Forest</strong>
                · R² {r2:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------
# DATA
# ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_data(n: int = 5000) -> pd.DataFrame:
    """Generate a synthetic household water-usage dataset.

    Uses a fixed random seed so results are reproducible across reruns
    (important since Streamlit re-executes the script on every interaction).
    """
    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(
        start="2024-01-01",
        periods=730,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "date": rng.choice(dates, n),
            "household_id": rng.integers(1, 501, n),
            "city": rng.choice(
                [
                    "Delhi",
                    "Mumbai",
                    "Bengaluru",
                    "Jaipur",
                    "Chennai",
                    "Hyderabad",
                    "Ahmedabad",
                ],
                n,
            ),
            "household_size": rng.integers(2, 8, n),
            "temperature_c": rng.normal(29, 6, n).round(1),
            "rainfall_mm": rng.gamma(2, 4, n).round(1),
            "water_supply_lpd": np.clip(rng.normal(650, 100, n), 350, 900).round(1),
            "previous_consumption_lpd": np.clip(
                rng.normal(500, 100, n), 250, 800
            ).round(1),
            "toilet_flushes": rng.integers(8, 30, n),
            "flush_litres": rng.choice([3, 4, 5, 6, 8, 10], n),
            "greywater_available_lpd": np.clip(
                rng.normal(100, 30, n), 20, 200
            ).round(1),
            "recycled_water_lpd": np.clip(
                rng.normal(70, 25, n), 0, 150
            ).round(1),
            "leakage_lpd": np.clip(rng.exponential(15, n), 0, 100).round(1),
            "awareness_score": rng.integers(30, 101, n),
            "water_stress_index": rng.uniform(0.2, 0.95, n).round(3),
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    season_factor = np.where(
        df["month"].isin([4, 5, 6]),
        1.12,
        np.where(df["month"].isin([7, 8, 9]), 0.92, 1.0),
    )

    base = (
        df["household_size"] * 90
        + df["temperature_c"] * 3
        + df["previous_consumption_lpd"] * 0.35
        + df["toilet_flushes"] * df["flush_litres"] * 0.35
        + df["leakage_lpd"] * 0.5
        - df["rainfall_mm"] * 0.7
        - df["awareness_score"] * 0.35
        - df["recycled_water_lpd"] * 0.25
    )

    noise = rng.normal(0, 35, n)
    df[TARGET] = (base * season_factor + noise).clip(150, 1200).round(2)

    return df.drop(columns=["month"]).sort_values("date").reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features (calendar parts, totals, balances) from raw data."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise ValueError("No valid dates were found in the dataset.")

    df["date"] = df["date"].fillna(df["date"].median())

    numeric = df.select_dtypes(include=np.number).columns
    if len(numeric):
        df[numeric] = df[numeric].fillna(df[numeric].median())

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df["toilet_water_lpd"] = df["toilet_flushes"] * df["flush_litres"]
    df["total_available_lpd"] = (
        df["water_supply_lpd"] + df["recycled_water_lpd"]
    )

    if TARGET in df.columns:
        df["water_balance_lpd"] = (
            df["total_available_lpd"] - df[TARGET]
        )

    return df


# ----------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train the Random Forest demand model and return it with evaluation metrics."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    evaluation = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": prediction,
        }
    )

    return model, mae, rmse, r2, importance, evaluation


def add_predictions(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """Attach AI predictions, water balance, risk tier and recommendations to df."""
    df = df.copy()

    df["predicted_demand_lpd"] = model.predict(df[FEATURES])

    df["total_available_lpd"] = (
        df["water_supply_lpd"] + df["recycled_water_lpd"]
    )

    df["water_balance_lpd"] = (
        df["total_available_lpd"] - df["predicted_demand_lpd"]
    )

    df["shortage_risk"] = np.select(
        [
            df["water_balance_lpd"] < -100,
            df["water_balance_lpd"] < 0,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    def recommendation(row):
        if row["shortage_risk"] == "HIGH":
            return (
                "Reduce freshwater use, increase recycled-water use, "
                "check leakage and adopt efficient flushing."
            )
        if row["shortage_risk"] == "MEDIUM":
            return (
                "Monitor consumption, reuse greywater and reduce "
                "toilet flush volume."
            )
        return "Maintain efficient consumption and monitor water balance."

    df["recommendation"] = df.apply(recommendation, axis=1)

    df["current_flush_water"] = df["toilet_flushes"] * BASELINE_FLUSH_LITRES
    df["efficient_flush_water"] = df["toilet_flushes"] * EFFICIENT_FLUSH_LITRES
    df["potential_flush_saving_lpd"] = (
        df["current_flush_water"] - df["efficient_flush_water"]
    ).clip(lower=0)
    df["annual_flush_saving_litres"] = (
        df["potential_flush_saving_lpd"] * 365
    )

    return df


@st.cache_data(show_spinner=False)
def build_dataset():
    return prepare_features(generate_data())


@st.cache_resource(show_spinner=False)
def build_ai():
    prepared = build_dataset()

    model, mae, rmse, r2, importance, evaluation = train_model(prepared)
    predictions = add_predictions(prepared, model)

    # Persist artifacts to disk when possible, but the app must never depend
    # on this succeeding (e.g. read-only filesystems, disk quota issues).
    if _DATA_DIR_WRITABLE:
        try:
            predictions.to_csv(PREDICTIONS_FILE, index=False)
            generate_data().to_csv(DATA_FILE, index=False)
            joblib.dump(model, MODEL_FILE)
        except OSError:
            pass

    return predictions, model, mae, rmse, r2, importance, evaluation


# ----------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------
with st.spinner("Initializing AquaAI AI engine..."):
    df, model, mae, rmse, r2, importance, evaluation = build_ai()

inject_css(st.session_state.dark_mode)


# ----------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------
NAV_ITEMS = [
    ("🏠 Overview", "Overview"),
    ("🤖 AI Predictions", "AI Predictions"),
    ("🔮 Forecast", "Forecast"),
    ("🚨 Risk Monitor", "Risk Monitor"),
    ("🏙️ City Analytics", "City Analytics"),
    ("🏠 Household Simulator", "Household Simulator"),
    ("💧 Leakage Intelligence", "Leakage Intelligence"),
    ("♻️ Sustainability", "Sustainability"),
    ("🧠 Explainable AI", "Explainable AI"),
    ("📑 Reports", "Reports"),
]

with st.sidebar:
    st.markdown(
        """
        <div class="aqua-brand">
            <div class="aqua-drop">💧</div>
            <div class="aqua-name">AquaAI</div>
            <div class="aqua-tagline">SMART WATER INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-title">MAIN DASHBOARD</div>', unsafe_allow_html=True)

    for index, (page_name, _) in enumerate(NAV_ITEMS):
        active = st.session_state.aqua_navigation == page_name

        if active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)

        clicked = st.button(
            page_name,
            key=f"aqua_nav_button_{index}",
            use_container_width=True,
        )

        if active:
            st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state.aqua_navigation = page_name
            st.rerun()

    st.divider()

    dark_value = st.toggle(
        "🌙 Dark mode",
        value=st.session_state.dark_mode,
        key="sidebar_dark_mode",
    )

    if dark_value != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_value
        st.rerun()

    high_risk_count = int((df["shortage_risk"] == "HIGH").sum())

    st.markdown(
        f"""
        <div class="sidebar-status">
            <b style="color:white;">🟢 SYSTEM ONLINE</b>
            <br><br>
            🤖 AI Model: Random Forest<br>
            📊 Records: {len(df):,}<br>
            🎯 R² Score: {r2:.3f}<br>
            🚨 High Risk: {high_risk_count:,}<br>
            💧 Water Engine: Active
        </div>
        <div class="sidebar-footer">
            AquaAI v2.0<br>
            AI • Analytics • Sustainability
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.aqua_navigation


# =================================================================
# 1. OVERVIEW
# =================================================================
if page == "🏠 Overview":
    hero(
        "AquaAI",
        "AI-powered household water intelligence. Predict demand, "
        "monitor shortage risk and discover water-saving opportunities.",
    )

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            """
            <style>
            .get-started-wrap .stButton > button {
                min-height: 58px !important;
                border-radius: 999px !important;
                font-size: 20px !important;
                font-weight: 800 !important;
                background: linear-gradient(135deg,#0799f5,#276cf5,#7b2ff7) !important;
                box-shadow: 0 14px 32px rgba(43,108,245,.28) !important;
            }
            </style>
            <div class="get-started-wrap">
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Get Started",
            key="overview_get_started",
            use_container_width=True,
        ):
            st.session_state.aqua_navigation = "🤖 AI Predictions"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    dashboard_status(len(df), r2, high_risk_count)

    metric_cards(
        [
            ("Households", f"{df['household_id'].nunique():,}"),
            ("Avg AI Demand", f"{df['predicted_demand_lpd'].mean():.0f} L/day"),
            ("Avg Available", f"{df['total_available_lpd'].mean():.0f} L/day"),
            ("Saving Potential", f"{df['potential_flush_saving_lpd'].mean():,.0f} L/day"),
            ("High Risk Records", f"{high_risk_count:,}"),
        ]
    )

    section("📈 Water Demand Trend")
    daily = df.groupby("date")[
        ["actual_consumption_lpd", "predicted_demand_lpd", "total_available_lpd"]
    ].mean()

    st.line_chart(
        daily.rename(
            columns={
                "actual_consumption_lpd": "Actual",
                "predicted_demand_lpd": "AI Prediction",
                "total_available_lpd": "Available",
            }
        )
    )

    section("🚨 Risk Distribution")
    risk_counts = (
        df["shortage_risk"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"])
        .fillna(0)
    )
    st.bar_chart(risk_counts)

    section("🏙️ City Snapshot")
    city_snapshot = (
        df.groupby("city")
        .agg(
            demand_lpd=("predicted_demand_lpd", "mean"),
            available_lpd=("total_available_lpd", "mean"),
            leakage_lpd=("leakage_lpd", "mean"),
        )
        .round(1)
    )
    st.dataframe(city_snapshot, use_container_width=True)


# =================================================================
# 2. AI PREDICTIONS
# =================================================================
elif page == "🤖 AI Predictions":
    hero(
        "AI Predictions",
        "Random Forest-powered water-demand prediction.",
    )

    metric_cards(
        [
            ("R² Score", f"{r2:.3f}"),
            ("MAE", f"{mae:.1f} L/day"),
            ("RMSE", f"{rmse:.1f} L/day"),
        ]
    )

    section("📊 Actual vs Predicted")
    st.scatter_chart(evaluation, x="actual", y="predicted")

    section("🧠 Feature Importance")
    st.dataframe(importance, use_container_width=True, hide_index=True)

    section("📋 Prediction Sample")
    columns = [
        "date",
        "household_id",
        "city",
        "actual_consumption_lpd",
        "predicted_demand_lpd",
        "total_available_lpd",
        "water_balance_lpd",
        "shortage_risk",
    ]
    st.dataframe(df[columns].head(100), use_container_width=True, hide_index=True)


# =================================================================
# 3. FORECAST
# =================================================================
elif page == "🔮 Forecast":
    hero(
        "Forecast",
        "Estimate upcoming household water demand using recent "
        "AI-predicted demand patterns.",
    )

    days = st.selectbox("Forecast horizon", [7, 30, 90])

    daily = (
        df.groupby("date")["predicted_demand_lpd"]
        .mean()
        .sort_index()
    )
    last_14 = daily.tail(14)

    if len(last_14) >= 2:
        x = np.arange(len(last_14))
        trend = float(np.polyfit(x, last_14.to_numpy(), 1)[0])
    else:
        trend = 0.0

    baseline = float(last_14.mean()) if len(last_14) else float(daily.mean())

    future_dates = pd.date_range(
        daily.index.max() + pd.Timedelta(days=1),
        periods=days,
    )

    forecast = np.maximum(
        150,
        baseline + trend * np.arange(1, days + 1),
    )

    forecast_df = pd.DataFrame(
        {"forecast_lpd": forecast},
        index=future_dates,
    )

    metric_cards(
        [
            ("Forecast Days", str(days)),
            ("Starting Level", f"{baseline:.0f} L/day"),
            ("Average Forecast", f"{forecast.mean():.0f} L/day"),
        ]
    )

    section("🔮 Forecast Trend")
    st.line_chart(forecast_df)

    st.dataframe(forecast_df.round(1), use_container_width=True)

    st.info(
        "This is a trend-based projection from recent AI predictions; "
        "it is not a separately trained time-series forecasting model."
    )


# =================================================================
# 4. RISK MONITOR
# =================================================================
elif page == "🚨 Risk Monitor":
    hero(
        "Risk Monitor",
        "Identify records where predicted demand is higher than available water.",
    )

    counts = (
        df["shortage_risk"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"])
        .fillna(0)
    )

    metric_cards(
        [
            ("LOW", f"{int(counts['LOW']):,}"),
            ("MEDIUM", f"{int(counts['MEDIUM']):,}"),
            ("HIGH", f"{int(counts['HIGH']):,}"),
        ]
    )

    section("🚨 Risk Distribution")
    st.bar_chart(counts)

    section("🔥 Priority Households")
    priority = (
        df[df["shortage_risk"] == "HIGH"]
        .sort_values("water_balance_lpd")
        .head(100)
    )

    priority_columns = [
        "household_id",
        "city",
        "household_size",
        "predicted_demand_lpd",
        "total_available_lpd",
        "water_balance_lpd",
        "leakage_lpd",
        "recommendation",
    ]
    st.dataframe(
        priority[priority_columns],
        use_container_width=True,
        hide_index=True,
    )


# =================================================================
# 5. CITY ANALYTICS
# =================================================================
elif page == "🏙️ City Analytics":
    hero(
        "City Analytics",
        "Compare demand, supply, leakage and recycling patterns across cities.",
    )

    city = st.selectbox(
        "Select city",
        ["All"] + sorted(df["city"].unique().tolist()),
    )

    view = df if city == "All" else df[df["city"] == city]

    metric_cards(
        [
            ("Predicted Demand", f"{view['predicted_demand_lpd'].mean():.0f} L/day"),
            ("Available Water", f"{view['total_available_lpd'].mean():.0f} L/day"),
            ("Leakage", f"{view['leakage_lpd'].mean():.1f} L/day"),
            ("Recycled", f"{view['recycled_water_lpd'].mean():.0f} L/day"),
        ]
    )

    section("🏙️ Average Demand by City")
    city_demand = (
        df.groupby("city")["predicted_demand_lpd"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(city_demand)

    section("📊 City Summary")
    summary = (
        df.groupby("city")
        .agg(
            households=("household_id", "nunique"),
            demand_lpd=("predicted_demand_lpd", "mean"),
            available_lpd=("total_available_lpd", "mean"),
            leakage_lpd=("leakage_lpd", "mean"),
            recycled_lpd=("recycled_water_lpd", "mean"),
            high_risk_records=(
                "shortage_risk",
                lambda x: int((x == "HIGH").sum()),
            ),
        )
        .round(1)
    )
    st.dataframe(summary, use_container_width=True)


# =================================================================
# 6. HOUSEHOLD SIMULATOR
# =================================================================
elif page == "🏠 Household Simulator":
    hero(
        "Household Simulator",
        "Change household conditions and instantly estimate daily water demand and shortage risk.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        size = st.slider("Household size", 2, 12, 4)
        temp = st.slider("Temperature °C", 10, 50, 30)
        rainfall = st.slider("Rainfall mm", 0.0, 100.0, 5.0)
        supply = st.slider("Water supply L/day", 200, 1200, 650)
        previous = st.slider("Previous consumption L/day", 150, 1200, 500)

    with c2:
        flushes = st.slider("Toilet flushes/day", 1, 40, 18)
        flush = st.select_slider(
            "Flush litres",
            options=[3, 4, 5, 6, 8, 10],
            value=6,
        )
        grey = st.slider("Greywater available L/day", 0, 250, 100)
        recycled = st.slider("Recycled water L/day", 0, 200, 70)

    with c3:
        leakage = st.slider("Leakage L/day", 0.0, 100.0, 10.0)
        awareness = st.slider("Awareness score", 0, 100, 75)
        stress = st.slider("Water stress index", 0.0, 1.0, 0.5)

    today = pd.Timestamp.today()

    row = pd.DataFrame(
        [{
            "household_size": size,
            "temperature_c": temp,
            "rainfall_mm": rainfall,
            "water_supply_lpd": supply,
            "previous_consumption_lpd": previous,
            "toilet_flushes": flushes,
            "flush_litres": flush,
            "greywater_available_lpd": grey,
            "recycled_water_lpd": recycled,
            "leakage_lpd": leakage,
            "awareness_score": awareness,
            "water_stress_index": stress,
            "month": today.month,
            "day_of_week": today.dayofweek,
        }]
    )

    try:
        prediction = float(model.predict(row[FEATURES])[0])
    except Exception:
        st.error("Could not compute a prediction for these inputs. Please adjust the sliders and try again.")
        st.stop()

    available = supply + recycled
    balance = available - prediction

    risk = (
        "HIGH" if balance < -100
        else "MEDIUM" if balance < 0
        else "LOW"
    )

    flush_saving = flushes * max(0, flush - EFFICIENT_FLUSH_LITRES)

    metric_cards(
        [
            ("Predicted Demand", f"{prediction:.0f} L/day"),
            ("Available", f"{available:.0f} L/day"),
            ("Balance", f"{balance:.0f} L/day"),
            ("Risk", risk),
            ("Potential Flush Saving", f"{flush_saving:.0f} L/day"),
        ]
    )

    if risk == "HIGH":
        st.error("🚨 High shortage risk. Reduce freshwater use and inspect leakage.")
    elif risk == "MEDIUM":
        st.warning("⚠️ Medium shortage risk. Reuse water and reduce consumption.")
    else:
        st.success("🟢 Low shortage risk under these conditions.")

    section("💡 Simulator Recommendation")

    recommendations = []
    if leakage > 25:
        recommendations.append("💧 Leakage is relatively high. Inspect taps, pipes and toilet systems.")
    if flush > 6:
        recommendations.append("🚽 Consider a lower-volume efficient flush.")
    if recycled < 50:
        recommendations.append("♻️ Increasing recycled-water use can reduce dependence on fresh supply.")
    if awareness < 60:
        recommendations.append("🧠 Improving water-use awareness may help reduce unnecessary consumption.")

    if recommendations:
        for item in recommendations:
            st.write(item)
    else:
        st.write("✅ Current settings show no additional simulator recommendation.")


# =================================================================
# 7. LEAKAGE INTELLIGENCE
# =================================================================
elif page == "💧 Leakage Intelligence":
    hero(
        "Leakage Intelligence",
        "Find where water losses are concentrated and estimate their impact.",
    )

    average_leakage = float(df["leakage_lpd"].mean())
    monthly_loss = average_leakage * 30
    annual_loss = average_leakage * 365

    metric_cards(
        [
            ("Average Leakage", f"{average_leakage:.1f} L/day"),
            ("Monthly Estimate", f"{monthly_loss:,.0f} L"),
            ("Annual Estimate", f"{annual_loss:,.0f} L"),
        ]
    )

    section("🚨 Highest Leakage Records")
    top_leakage = df.nlargest(50, "leakage_lpd")[
        [
            "household_id",
            "city",
            "leakage_lpd",
            "predicted_demand_lpd",
            "water_balance_lpd",
            "shortage_risk",
        ]
    ]
    st.dataframe(top_leakage, use_container_width=True, hide_index=True)

    section("📊 Leakage by City")
    leakage_city = (
        df.groupby("city")["leakage_lpd"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(leakage_city)

    section("🔎 Leakage Distribution")
    leakage_trend = (
        df.sort_values("date")
        .set_index("date")["leakage_lpd"]
        .rolling(30, min_periods=1)
        .mean()
    )
    st.line_chart(leakage_trend)


# =================================================================
# 8. SUSTAINABILITY
# =================================================================
elif page == "♻️ Sustainability":
    hero(
        "Sustainability",
        "Measure opportunities from efficient flushing, recycling, reuse and leakage reduction.",
    )

    annual_flush = float(df["annual_flush_saving_litres"].sum())
    recycled = float(df["recycled_water_lpd"].mean())
    flush_saving = float(df["potential_flush_saving_lpd"].mean())
    leakage = float(df["leakage_lpd"].mean())

    metric_cards(
        [
            ("Annual Flush Saving", f"{annual_flush:,.0f} L"),
            ("Average Recycled", f"{recycled:.0f} L/day"),
            ("Potential Flush Saving", f"{flush_saving:.0f} L/day"),
            ("Average Leakage", f"{leakage:.1f} L/day"),
        ]
    )

    section("♻️ Sustainability Opportunity")
    st.progress(
        min(1.0, max(0.0, flush_saving / 100)),
        text=f"Flush-efficiency opportunity: {flush_saving:.0f} L/day average",
    )

    section("🚽 Flush Efficiency")
    flush_table = (
        df.groupby("flush_litres")
        .agg(
            average_saving_lpd=("potential_flush_saving_lpd", "mean"),
            households=("household_id", "count"),
        )
        .round(1)
    )
    st.dataframe(flush_table, use_container_width=True)

    section("♻️ Recycled Water by City")
    recycled_city = (
        df.groupby("city")["recycled_water_lpd"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(recycled_city)

    st.info("These are demonstration estimates based on AquaAI's synthetic dataset.")


# =================================================================
# 9. EXPLAINABLE AI
# =================================================================
elif page == "🧠 Explainable AI":
    hero(
        "Explainable AI",
        "Understand which input variables are most influential in AquaAI's Random Forest model.",
    )

    section("🧠 Model Feature Importance")
    top_features = importance.head(14).set_index("feature")["importance"]
    st.bar_chart(top_features)

    st.dataframe(importance, use_container_width=True, hide_index=True)

    section("🔍 Explain a Household Prediction")

    index = st.number_input(
        "Select dataset row",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        step=1,
    )

    selected = df.iloc[int(index)]

    st.write(
        f"**Household:** {selected['household_id']}  |  "
        f"**City:** {selected['city']}"
    )

    importance_indexed = importance.set_index("feature")

    contribution_data = pd.DataFrame(
        {
            "feature": FEATURES,
            "value": [selected[f] for f in FEATURES],
            "importance": [
                importance_indexed.loc[f, "importance"]
                for f in FEATURES
            ],
        }
    ).sort_values("importance", ascending=False)

    st.dataframe(
        contribution_data,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Random Forest feature importance indicates relative model influence. "
        "It does not prove that a feature causes water consumption."
    )


# =================================================================
# 10. REPORTS
# =================================================================
elif page == "📑 Reports":
    hero(
        "Reports",
        "Export AquaAI predictions, source data and a compact management report.",
    )

    section("📊 CSV Downloads")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📊 Download Predictions CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="aquaai_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        raw_data = generate_data()
        st.download_button(
            "🗃️ Download Raw Dataset CSV",
            data=raw_data.to_csv(index=False).encode("utf-8"),
            file_name="aquaai_household_water_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    section("📄 PDF Report")

    if not REPORTLAB_AVAILABLE:
        st.warning(
            "ReportLab is not installed. Install it with: "
            "python3 -m pip install reportlab"
        )
    else:

        @st.cache_data(show_spinner=False)
        def create_pdf(
            record_count: int,
            household_count: int,
            avg_demand: float,
            avg_available: float,
            high_risk_count: int,
            r2_score_: float,
            mae_: float,
            rmse_: float,
            risk_counts: tuple,
        ) -> bytes:
            """Build the summary PDF report. Cached on its scalar inputs so it is
            only rebuilt when the underlying figures actually change, instead of
            on every Streamlit rerun."""
            buffer = BytesIO()

            document = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )

            styles = getSampleStyleSheet()

            story = [
                Paragraph("AquaAI Water Management Report", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"Records analyzed: {record_count:,}", styles["BodyText"]),
                Paragraph(f"Households: {household_count:,}", styles["BodyText"]),
                Paragraph(
                    f"Average predicted demand: {avg_demand:.1f} L/day",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"Average available water: {avg_available:.1f} L/day",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"High-risk records: {high_risk_count:,}",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"Model R²: {r2_score_:.3f} | MAE: {mae_:.1f} | RMSE: {rmse_:.1f}",
                    styles["BodyText"],
                ),
                Spacer(1, 15),
                Paragraph("Risk Summary", styles["Heading2"]),
            ]

            low_count, medium_count, high_count = risk_counts

            table_data = [
                ["Risk", "Records"],
                ["LOW", low_count],
                ["MEDIUM", medium_count],
                ["HIGH", high_count],
            ]

            table = Table(table_data, colWidths=[120, 100])

            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ]
                )
            )

            story.extend(
                [
                    table,
                    Spacer(1, 15),
                    Paragraph(
                        "AquaAI uses a synthetic demonstration dataset. "
                        "Results should not be treated as real-world measurements.",
                        styles["BodyText"],
                    ),
                ]
            )

            document.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        risk_value_counts = df["shortage_risk"].value_counts()

        pdf_bytes = create_pdf(
            record_count=len(df),
            household_count=int(df["household_id"].nunique()),
            avg_demand=float(df["predicted_demand_lpd"].mean()),
            avg_available=float(df["total_available_lpd"].mean()),
            high_risk_count=int((df["shortage_risk"] == "HIGH").sum()),
            r2_score_=float(r2),
            mae_=float(mae),
            rmse_=float(rmse),
            risk_counts=(
                int(risk_value_counts.get("LOW", 0)),
                int(risk_value_counts.get("MEDIUM", 0)),
                int(risk_value_counts.get("HIGH", 0)),
            ),
        )

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="AquaAI_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    section("📋 Current Model Summary")

    st.dataframe(
        pd.DataFrame(
            {
                "Metric": [
                    "Model",
                    "Trees",
                    "Max Depth",
                    "MAE",
                    "RMSE",
                    "R²",
                ],
                "Value": [
                    "Random Forest Regressor",
                    "300",
                    "15",
                    f"{mae:.2f}",
                    f"{rmse:.2f}",
                    f"{r2:.4f}",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------
st.markdown(
    """
    <br><br>
    <div style="
        text-align:center;
        opacity:.65;
        padding:20px;
        font-size:13px;
    ">
        💧 <b>AquaAI</b> — Smart Water Management Dashboard
        <br>
        AI • Analytics • Sustainability • Water Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
