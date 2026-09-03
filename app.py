import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json
import math
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SolarGen AI | Solar Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. MODEL & METRICS LOADING (WITH CACHING & SAFE FALLBACKS)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" if (BASE_DIR / "models").exists() else BASE_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "solar_power_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading machine learning model: {e}")
        return None

@st.cache_data
def load_metrics(path: Path):
    if not path.exists():
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

# ============================================================
# 3. UI / UX CSS
# ============================================================
st.markdown("""
<style>
.stApp { background:#f4f8fc; color:#14253d; }
.main .block-container { max-width:1180px; padding:34px 30px 55px; }
h1,h2,h3 { color:#14253d !important; }
p { color:#60748a; }
hr { border:0; border-top:1px solid #dce5ee; margin:30px 0; }

.brand-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:34px; }
.brand-left { display:flex; align-items:center; gap:11px; }
.brand-icon { width:38px; height:38px; border-radius:10px; background:#1769ff; display:flex; align-items:center; justify-content:center; font-size:20px; }
.brand-name { color:#14253d; font-size:17px; font-weight:800; }
.live-badge { display:flex; align-items:center; gap:7px; padding:7px 11px; border-radius:20px; background:#e8f8ef; color:#16834a; font-size:11px; font-weight:800; }
.live-dot { width:7px; height:7px; border-radius:50%; background:#19a85b; }

.hero-label { color:#1769ff; font-size:11px; font-weight:800; letter-spacing:1.2px; margin-bottom:10px; }
.hero-title { color:#14253d; font-size:42px; line-height:1.12; font-weight:850; letter-spacing:-1px; }
.hero-description { color:#6b7f95; font-size:15px; line-height:1.6; max-width:720px; margin:10px 0 35px; }

.section-title { color:#14253d; font-size:22px; font-weight:800; margin-top:12px; margin-bottom:5px; }
.section-subtitle { color:#718399; font-size:13px; margin-bottom:18px; }

div[data-testid="stWidgetLabel"] p,
label,
[data-testid="stWidgetLabel"] { color:#203650 !important; font-size:13px !important; font-weight:750 !important; }

div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stTimeInput"] div[data-baseweb="input"],
div[data-testid="stDateInput"] div[data-baseweb="input"] { background:#fff !important; border:1px solid #d8e2ec !important; border-radius:10px !important; min-height:44px; box-shadow:0 2px 7px rgba(20,37,61,.035); }
div[data-testid="stNumberInput"] input, input { background:#fff !important; color:#14253d !important; -webkit-text-fill-color:#14253d !important; font-weight:650 !important; font-size:14px !important; }
div[data-testid="stNumberInput"] button { background:#fff !important; color:#60748a !important; border:0 !important; }

.kpi-card { background:#fff; border:1px solid #e0e8f0; border-radius:14px; padding:20px 21px; min-height:116px; box-shadow:0 5px 18px rgba(20,37,61,.045); }
.kpi-card-blue { background:#1769ff; border-color:#1769ff; }
.kpi-label { color:#7a8da2; font-size:10px; font-weight:800; letter-spacing:.9px; margin-bottom:10px; }
.kpi-card-blue .kpi-label { color:#dce9ff; }
.kpi-value { color:#14253d; font-size:28px; font-weight:850; line-height:1.1; }
.kpi-card-blue .kpi-value { color:#fff; }
.kpi-note { color:#18a566; font-size:11px; margin-top:9px; }
.kpi-card-blue .kpi-note { color:#e7f0ff; }
.kpi-note-muted { color:#718399; font-size:11px; margin-top:9px; }

.chart-card { background:#fff; border:1px solid #e0e8f0; border-radius:16px; padding:20px 22px 4px; box-shadow:0 5px 18px rgba(20,37,61,.045); }
.chart-header { display:flex; justify-content:space-between; align-items:flex-start; }
.chart-title { color:#14253d; font-size:17px; font-weight:800; }
.chart-caption { color:#8a9aae; font-size:11px; margin-top:4px; }
.peak-badge { background:#fff5d9; color:#a87500; border-radius:18px; padding:7px 11px; font-size:10px; font-weight:800; white-space:nowrap; }

.status-strip { background:#edf5ff; border:1px solid #d7e8ff; color:#3c6189; border-radius:10px; padding:11px 14px; font-size:11px; margin-top:12px; }
div[data-testid="stDataFrame"] { border:1px solid #dfe7ef; border-radius:12px; overflow:hidden; }

@media (max-width:800px) {
    .hero-title { font-size:32px; }
    .main .block-container { padding-left:18px; padding-right:18px; }
}
</style>
""", unsafe_allow_html=True)

# Check model status
if model is None:
    st.warning("⚠️ Model file `solar_power_model.pkl` not found in `/models`. Please place your trained `.pkl` file in the `models/` directory.")

# ============================================================
# 4. HEADER + HERO
# ============================================================
now = datetime.now()
today_str = now.strftime("%d %b %Y")

st.markdown(f"""
<div class="brand-row">
  <div class="brand-left">
    <div class="brand-icon">☀️</div>
    <div class="brand-name">SolarGen AI</div>
  </div>
  <div class="live-badge"><span class="live-dot"></span> LIVE MODEL</div>
</div>

<div class="hero-label">GENERATION FORECAST • {today_str.upper()}</div>
<div class="hero-title">Solar output, made predictable.</div>
<div class="hero-description">
  AI-powered solar generation forecasting using environmental conditions and time-based features.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. INPUTS (INTERACTIVE CONTROLS)
# ============================================================
st.markdown('<div class="section-title">Solar & Weather Conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Enter the current conditions used by the prediction model.</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    irradiance = st.number_input("Solar Irradiance (W/m²)", 0.0, 1200.0, 800.0, 10.0, help="Solar energy received per square metre.")
with c2:
    temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 25.0, 0.5, help="Current ambient temperature.")
with c3:
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 2.0, 0.5, help="Current wind speed.")

st.markdown('<div class="section-title" style="margin-top:20px;">Time & Sun Parameters</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Adjust time and solar position inputs for simulation.</div>', unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    selected_date = st.date_input("Simulation Date", value=now.date(), help="Select date for calculation.")
    day_of_year = selected_date.timetuple().tm_yday
    month = selected_date.month
with c5:
    selected_time = st.time_input("Simulation Hour", value=now.time().replace(minute=0, second=0), help="Select local 24-hour time.")
    hour = selected_time.hour
with c6:
    estimated_sun = max(0.0, round(90 * math.sin(math.pi * (hour - 6) / 12), 1)) if 6 <= hour <= 18 else 0.0
    sun_height = st.number_input("Sun Height (°)", 0.0, 90.0, float(estimated_sun), 1.0, help="Angle of the sun above horizon.")

# ============================================================
# 6. CURRENT PREDICTION
# ============================================================
input_data = pd.DataFrame({
    "irradiance": [irradiance],
    "temperature": [temperature],
    "wind_speed": [wind_speed],
    "day_of_year": [day_of_year],
    "hour": [hour],
    "sun_height": [sun_height],
    "month": [month],
})

if model is not None:
    if hasattr(model, "feature_names_in_"):
        input_data = input_data[model.feature_names_in_]
    prediction = max(0.0, float(model.predict(input_data)[0]))
else:
    prediction = 0.0

if sun_height <= 0 or irradiance <= 0:
    prediction = 0.0

predicted_kw = prediction / 1000

if irradiance >= 700:
    condition_score, condition_text = 95, "Excellent solar conditions"
elif irradiance >= 500:
    condition_score, condition_text = 85, "Good solar conditions"
elif irradiance >= 300:
    condition_score, condition_text = 70, "Moderate solar conditions"
elif irradiance >= 100:
    condition_score, condition_text = 50, "Low solar conditions"
else:
    condition_score, condition_text = 20, "Very low solar conditions"

# ============================================================
# 7. KPI CARDS
# ============================================================
st.markdown('<div class="section-title">Current Generation Overview</div>', unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">EXPECTED GENERATION</div>
      <div class="kpi-value">{prediction:,.1f} W</div>
      <div class="kpi-note">≈ {predicted_kw:.2f} kW predicted output</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">SOLAR CONDITION</div>
      <div class="kpi-value">{condition_score}%</div>
      <div class="kpi-note-muted">{condition_text}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    status_str = "ACTIVE" if model is not None else "DEMO MODE"
    st.markdown(f"""
    <div class="kpi-card kpi-card-blue">
      <div class="kpi-label">CURRENT STATUS</div>
      <div class="kpi-value">{status_str}</div>
      <div class="kpi-note">Random Forest model ready for forecasting</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 8. NEXT 6 HOURS FORECAST
# ============================================================
forecast_hours = []
forecast_power = []

base_dt = datetime.combine(selected_date, selected_time)

for i in range(1, 7):
    future_dt = base_dt + timedelta(hours=i)
    future_hour = future_dt.hour
    future_day = future_dt.timetuple().tm_yday
    future_month = future_dt.month

    if 6 <= future_hour <= 18:
        future_sun_height = max(0.0, 90 * math.sin(math.pi * (future_hour - 6) / 12))
    else:
        future_sun_height = 0.0

    if future_sun_height > 0 and sun_height > 0:
        future_irradiance = irradiance * (future_sun_height / max(sun_height, 10.0))
    elif future_sun_height > 0:
        future_irradiance = (future_sun_height / 90.0) * 800.0
    else:
        future_irradiance = 0.0

    future_irradiance = max(0.0, min(1200.0, future_irradiance))

    future_input = pd.DataFrame({
        "irradiance": [future_irradiance],
        "temperature": [temperature],
        "wind_speed": [wind_speed],
        "day_of_year": [future_day],
        "hour": [future_hour],
        "sun_height": [future_sun_height],
        "month": [future_month],
    })

    if model is not None:
        if hasattr(model, "feature_names_in_"):
            future_input = future_input[model.feature_names_in_]
        future_prediction = max(0.0, float(model.predict(future_input)[0]))
    else:
        future_prediction = 0.0

    if future_sun_height <= 0 or future_irradiance <= 0:
        future_prediction = 0.0

    forecast_hours.append(future_hour)
    forecast_power.append(future_prediction)

forecast_data = pd.DataFrame({"Hour": forecast_hours, "Predicted Power (W)": forecast_power})

peak_index = forecast_data["Predicted Power (W)"].idxmax()
peak_hour = int(forecast_data.loc[peak_index, "Hour"])
peak_power = float(forecast_data.loc[peak_index, "Predicted Power (W)"])
average_power = float(forecast_data["Predicted Power (W)"].mean())

# ============================================================
# 9. FORECAST GRAPH
# ============================================================
st.markdown('<div class="section-title" style="margin-top:35px;">Forecast trajectory</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Projected solar generation for the next six hours. Hover over any point for exact values.</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="chart-card">
  <div class="chart-header">
    <div>
      <div class="chart-title">Projected production profile</div>
      <div class="chart-caption">Hourly predicted solar power • Watts (W)</div>
    </div>
    <div class="peak-badge">☀ PEAK {peak_power:,.0f} W</div>
  </div>
</div>
""", unsafe_allow_html=True)

labels = [f"{h:02d}:00" for h in forecast_data["Hour"]]
peak_label = f"{peak_hour:02d}:00"

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=labels,
    y=forecast_data["Predicted Power (W)"],
    mode="lines+markers",
    name="Predicted solar power",
    line=dict(color="#1769FF", width=4, shape="spline"),
    marker=dict(size=9, color="#1769FF", line=dict(color="white", width=2)),
    fill="tozeroy",
    fillcolor="rgba(23,105,255,0.10)",
    hovertemplate="<b>%{x}</b><br>Predicted power: <b>%{y:,.1f} W</b><extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[peak_label],
    y=[peak_power],
    mode="markers+text",
    name="Peak generation",
    marker=dict(size=15, color="#F5B51B", line=dict(color="white", width=3)),
    text=[f"Peak {peak_power:,.0f} W"],
    textposition="top center",
    hovertemplate="<b>Peak generation</b><br>%{x}<br>Power: <b>%{y:,.1f} W</b><extra></extra>",
))
fig.update_layout(
    height=430,
    margin=dict(l=55, r=25, t=25, b=60),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial, sans-serif", color="#14253D", size=12),
    hovermode="x unified",
    showlegend=False,
    xaxis=dict(title="Time", showgrid=False, tickmode="linear"),
    yaxis=dict(title="Predicted Solar Power (W)", rangemode="tozero", gridcolor="#E7EDF4", zeroline=False),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "responsive": True})

st.markdown("""
<div class="status-strip">
  ↗ Forecast updates automatically when you change environmental inputs or simulation time.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 10. FORECAST SUMMARY & TABLE
# ============================================================
st.markdown('<div class="section-title" style="margin-top:30px;">Forecast summary</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Peak Power", f"{peak_power:,.1f} W")
with s2:
    st.metric("Peak Hour", f"{peak_hour:02d}:00")
with s3:
    st.metric("6-Hour Average", f"{average_power:,.1f} W")

st.markdown('<div class="section-title" style="margin-top:30px;">Hour-by-Hour Forecast</div>', unsafe_allow_html=True)
display_forecast = forecast_data.copy()
display_forecast["Hour"] = display_forecast["Hour"].apply(lambda x: f"{x:02d}:00")
display_forecast["Predicted Power (W)"] = display_forecast["Predicted Power (W)"].round(2)
display_forecast = display_forecast.rename(columns={"Hour": "Time", "Predicted Power (W)": "Predicted Solar Power (W)"})
st.dataframe(display_forecast, use_container_width=True, hide_index=True)

# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================
st.markdown('<div class="section-title" style="margin-top:35px;">Model performance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Evaluation metrics from the trained Random Forest Regression model.</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("MAE", f"{metrics.get('mae', 'N/A')} W", help="Mean Absolute Error.")
with m2:
    st.metric("RMSE", f"{metrics.get('rmse', 'N/A')} W", help="Root Mean Squared Error.")
with m3:
    r2_val = metrics.get('r2', 0.0)
    st.metric("R² Score", f"{r2_val:.4f}" if isinstance(r2_val, float) else str(r2_val), help="Goodness of fit metric.")

st.caption("Machine Learning Model: Random Forest Regressor")

# ============================================================
# 12. FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center;color:#8797aa;font-size:12px;padding:35px 0 10px;">
  SolarGen AI • Random Forest Regression • AI/ML-Based Solar Power Forecasting
</div>
""", unsafe_allow_html=True)
