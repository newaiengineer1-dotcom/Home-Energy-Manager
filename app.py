"""
Home Energy Management Dashboard
LESCO Protected Consumer | Captive Solar (No Export) | Sep 2026

Data window: Oct 2025 -> Sep 2026 (12 months)
Modern graphics. High-contrast theme.
"""
import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

# =============================================================
# SECRETS BRIDGE
# =============================================================
try:
    for _k in ("GROQ_API_KEY",):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k]).strip()
except Exception:
    pass

# =============================================================
# CONFIG
# =============================================================
DEFAULT_SOLAR_KWP = 5.0
DEFAULT_BATTERY_KWH = 10.0
PROTECTED_LIMIT = 200.0

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Data window: Oct 2025 -> Sep 2026 (12 months)
MONTH_SEQ = [
    "Oct-25", "Nov-25", "Dec-25",
    "Jan-26", "Feb-26", "Mar-26",
    "Apr-26", "May-26", "Jun-26",
    "Jul-26", "Aug-26", "Sep-26",
]

COLORS = {
    "bg":     "#0D1117",
    "bg2":    "#161B22",
    "border": "#30363D",
    "text":   "#F0F6FC",
    "muted":  "#8B949E",
    "primary":"#58A6FF",
    "success":"#3FB950",
    "warning":"#D29922",
    "danger": "#F85149",
    "solar":  "#FFDF4A",
    "battery":"#39D353",
    "grid":   "#58A6FF",
}

DEFAULT_APPLIANCES = [
    {"name": "Inverter AC 1.5T @26C (2x/week, 6h)", "watts": 550,  "hours": 6.0,  "days_per_week": 2, "qty": 1},
    {"name": "Inverter AC 1.5T @26C (daily, 16h)",  "watts": 550,  "hours": 16.0, "days_per_week": 7, "qty": 1},
    {"name": "Inverter AC 1.0T @26C (daily, 16h)",  "watts": 420,  "hours": 16.0, "days_per_week": 7, "qty": 1},
    {"name": "Microwave Oven (15 min/day)",         "watts": 1200, "hours": 0.25, "days_per_week": 7, "qty": 1},
    {"name": "Washing Machine (4h/week)",           "watts": 500,  "hours": 4.0,  "days_per_week": 1, "qty": 1},
    {"name": "Submersible Pump (2h/day)",           "watts": 750,  "hours": 2.0,  "days_per_week": 7, "qty": 1},
    {"name": "Machine 3kW (5h, 2x/week)",           "watts": 3000, "hours": 5.0,  "days_per_week": 2, "qty": 1},
    {"name": "RO Plant 1.5kW (5h, 2x/week)",        "watts": 1500, "hours": 5.0,  "days_per_week": 2, "qty": 1},
    {"name": "Ceiling Fan (BLDC)",                  "watts": 28,   "hours": 12.0, "days_per_week": 7, "qty": 6},
    {"name": "LED Light 12W",                       "watts": 12,   "hours": 6.0,  "days_per_week": 7, "qty": 30},
    {"name": "Philips Iron (3h/week)",              "watts": 1000, "hours": 3.0,  "days_per_week": 1, "qty": 1},
]

# =============================================================
# MODELS
# =============================================================
@dataclass
class Appliance:
    name: str
    watts: float
    hours: float
    days_per_week: int = 7
    qty: int = 1

    @property
    def monthly_kwh(self) -> float:
        weekly = (self.watts * self.hours * self.qty / 1000.0) * self.days_per_week
        return weekly * 4.33


# =============================================================
# HELPERS
# =============================================================
def _parse_num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _clean_col(df, col):
    if col not in df.columns:
        return []
    return [x for x in (_parse_num(v) for v in df[col]) if x is not None]


def _fmt_appliance_table(appliances):
    lines = ["Name | Watts | Hrs/Session | Days/Week | Qty | kWh/Month"]
    for a in appliances:
        lines.append("{} | {} | {} | {} | {} | {:.1f}".format(
            a.name, a.watts, a.hours, a.days_per_week, a.qty, a.monthly_kwh))
    return "\n".join(lines)


def _fmt_history_table(df):
    lines = ["Month | Consumption(kWh) | Bill(PKR)"]
    has = False
    for _, row in df.iterrows():
        cons = _parse_num(row.get("Consumption (kWh)"))
        bill = _parse_num(row.get("Bill (PKR)"))
        if cons is None and bill is None:
            continue
        has = True
        lines.append("{} | {} | {}".format(
            row["Month"],
            "{:.0f}".format(cons) if cons is not None else "N/A",
            "{:.0f}".format(bill) if bill is not None else "N/A",
        ))
    return "\n".join(lines) if has else "(no history provided)"


# =============================================================
# MODERN CHART BUILDERS
# =============================================================
def _modern_axis_style():
    return dict(
        showgrid=True,
        gridcolor="rgba(48,54,61,0.5)",
        gridwidth=1,
        zeroline=False,
        tickfont=dict(color=COLORS["text"], size=11),
        title_font=dict(color=COLORS["muted"], size=12),
        linecolor=COLORS["border"],
    )


def _base_layout(height=380, title=None):
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="-apple-system, sans-serif"),
        margin=dict(l=20, r=20, t=50 if title else 25, b=20),
        hoverlabel=dict(
            bgcolor=COLORS["bg2"],
            bordercolor=COLORS["primary"],
            font=dict(color=COLORS["text"], size=12),
        ),
    )
    if title:
        layout["title"] = dict(
            text=title, font=dict(color=COLORS["text"], size=15, weight="bold"),
            x=0.0, xanchor="left",
        )
    return layout


def chart_gauge(value, limit):
    """Modern gauge with gradient-style steps and clear threshold."""
    gauge_max = max(300, limit * 1.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={
            "suffix": " units",
            "font": {"size": 44, "color": COLORS["primary"], "family": "-apple-system"},
        },
        delta={
            "reference": limit,
            "increasing": {"color": COLORS["danger"]},
            "decreasing": {"color": COLORS["success"]},
            "font": {"size": 16},
        },
        gauge={
            "axis": {
                "range": [0, gauge_max],
                "tickcolor": COLORS["muted"],
                "tickfont": {"size": 10, "color": COLORS["muted"]},
                "tickwidth": 1,
            },
            "bar": {"color": COLORS["primary"], "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, limit * 0.75],        "color": "rgba(63,185,80,0.18)"},
                {"range": [limit * 0.75, limit],    "color": "rgba(210,153,34,0.22)"},
                {"range": [limit, gauge_max],       "color": "rgba(248,81,73,0.22)"},
            ],
            "threshold": {
                "line": {"color": COLORS["danger"], "width": 3},
                "thickness": 0.85,
                "value": limit,
            },
        },
    ))
    fig.update_layout(**_base_layout(height=290))
    return fig


def chart_monthly_stacked(history_df):
    """Modern stacked bar: Solar Used vs LESCO Import per month."""
    df = history_df.copy()
    df["solar_used"] = pd.to_numeric(df.get("Solar Used (kWh)"), errors="coerce")
    df["lesco"] = pd.to_numeric(df.get("LESCO Import (kWh)"), errors="coerce")
    df["cons"] = pd.to_numeric(df.get("Consumption (kWh)"), errors="coerce")

    # Fallback: if user only entered consumption, split visually
    if df["solar_used"].isna().all() and df["lesco"].isna().all():
        df["lesco"] = df["cons"]
        df["solar_used"] = 0
    df = df.fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Month"], y=df["solar_used"],
        name="Solar Used",
        marker=dict(
            color=COLORS["solar"],
            line=dict(color="rgba(255,223,74,0.9)", width=0),
        ),
        hovertemplate="<b>%{x}</b><br>Solar Used: %{y:.0f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["Month"], y=df["lesco"],
        name="LESCO Import",
        marker=dict(
            color=COLORS["primary"],
            line=dict(color="rgba(88,166,255,0.9)", width=0),
        ),
        hovertemplate="<b>%{x}</b><br>LESCO Import: %{y:.0f} kWh<extra></extra>",
    ))
    fig.add_hline(
        y=PROTECTED_LIMIT,
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
        annotation_text="Protected Limit · 200 kWh",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["danger"], size=11),
    )
    layout = _base_layout(height=400, title="12-Month Energy Breakdown")
    layout["barmode"] = "stack"
    layout["bargap"] = 0.35
    layout["xaxis"] = _modern_axis_style()
    layout["yaxis"] = dict(**_modern_axis_style(), title="kWh")
    layout["legend"] = dict(
        orientation="h", y=1.12, x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=11),
    )
    fig.update_layout(**layout)
    return fig


def chart_solar_utilization(history_df):
    """Modern grouped bar: Generated vs Used."""
    df = history_df.copy()
    df["gen"] = pd.to_numeric(df.get("Solar Gen (kWh)"), errors="coerce")
    df["used"] = pd.to_numeric(df.get("Solar Used (kWh)"), errors="coerce")
    df = df.dropna(subset=["gen", "used"], how="all").fillna(0)
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Month"], y=df["gen"], name="Generated",
        marker=dict(color="rgba(255,223,74,0.35)",
                    line=dict(color=COLORS["solar"], width=1.5)),
        hovertemplate="<b>%{x}</b><br>Generated: %{y:.0f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["Month"], y=df["used"], name="Used On-site",
        marker=dict(color=COLORS["success"],
                    line=dict(color="rgba(63,185,80,0.9)", width=0)),
        hovertemplate="<b>%{x}</b><br>Used: %{y:.0f} kWh<extra></extra>",
    ))
    layout = _base_layout(height=340, title="Solar Generation vs Utilization")
    layout["barmode"] = "group"
    layout["bargap"] = 0.3
    layout["bargroupgap"] = 0.1
    layout["xaxis"] = _modern_axis_style()
    layout["yaxis"] = dict(**_modern_axis_style(), title="kWh")
    layout["legend"] = dict(
        orientation="h", y=1.12, x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=11),
    )
    fig.update_layout(**layout)
    return fig


def chart_bill_trend(history_df):
    """Modern line chart: Bill (PKR) with area fill."""
    df = history_df.copy()
    df["bill"] = pd.to_numeric(df.get("Bill (PKR)"), errors="coerce")
    df = df.dropna(subset=["bill"])
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Month"], y=df["bill"],
        mode="lines+markers",
        name="Bill",
        line=dict(color=COLORS["warning"], width=3, shape="spline", smoothing=0.9),
        marker=dict(size=9, color=COLORS["warning"],
                    line=dict(color=COLORS["text"], width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(210,153,34,0.15)",
        hovertemplate="<b>%{x}</b><br>Bill: PKR %{y:,.0f}<extra></extra>",
    ))
    layout = _base_layout(height=340, title="Monthly Bill Trend (PKR)")
    layout["xaxis"] = _modern_axis_style()
    layout["yaxis"] = dict(**_modern_axis_style(), title="PKR")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def chart_appliance_modern(app_df):
    """Modern donut with legend, no clutter."""
    labels = app_df["Appliance"].tolist()
    values = app_df["kWh/Month"].tolist()

    palette = [
        "#58A6FF", "#3FB950", "#FFDF4A", "#F85149", "#D29922",
        "#A371F7", "#39D353", "#FF7B72", "#79C0FF", "#F0883E",
        "#7EE787",
    ]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(
            colors=palette[:len(labels)],
            line=dict(color=COLORS["bg"], width=2),
        ),
        textinfo="percent",
        textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(color="#0D1117", size=11, family="-apple-system"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f} kWh/mo<br>%{percent}<extra></extra>",
        sort=True,
    ))
    layout = _base_layout(height=420, title="Appliance Monthly Consumption")
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="v",
        yanchor="middle", y=0.5,
        xanchor="left", x=1.02,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=11),
    )
    layout["margin"] = dict(l=10, r=10, t=50, b=10)
    fig.update_layout(**layout)
    return fig


def chart_consumption_line(history_df, appliance_estimate):
    """Modern smooth line of past consumption vs appliance estimate."""
    df = history_df.copy()
    df["cons"] = pd.to_numeric(df.get("Consumption (kWh)"), errors="coerce")
    df = df.dropna(subset=["cons"])
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Month"], y=df["cons"],
        mode="lines+markers",
        name="Metered Consumption",
        line=dict(color=COLORS["primary"], width=3,
                  shape="spline", smoothing=0.9),
        marker=dict(size=10, color=COLORS["primary"],
                    line=dict(color="#FFFFFF", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.12)",
        hovertemplate="<b>%{x}</b><br>Consumption: %{y:.0f} kWh<extra></extra>",
    ))
    fig.add_hline(
        y=appliance_estimate,
        line=dict(color=COLORS["warning"], width=2, dash="dot"),
        annotation_text="Appliance Est. · {:.0f} kWh".format(appliance_estimate),
        annotation_position="top right",
        annotation_font=dict(color=COLORS["warning"], size=11),
    )
    layout = _base_layout(height=340, title="Metered Consumption Trend")
    layout["xaxis"] = _modern_axis_style()
    layout["yaxis"] = dict(**_modern_axis_style(), title="kWh")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# =============================================================
# LLM CALL
# =============================================================
def call_llm_analysis(solar_kwp, battery_kwh, appliances_df, history_df):
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return {"error": "GROQ_API_KEY not configured in Streamlit Secrets."}

    appliance_table = _fmt_appliance_table(appliances_df)
    history_table = _fmt_history_table(history_df)
    appliance_monthly = sum(a.monthly_kwh for a in appliances_df)

    system_prompt = (
        "You are a senior solar energy engineer and energy advisor for Lahore, "
        "Pakistan. You know:\n"
        "- Lahore irradiance: ~5.2 peak sun hours/day annual average\n"
        "- Residential PV system losses: ~20%\n"
        "- LiFePO4 battery round-trip efficiency: ~85%\n"
        "- LESCO protected-consumer tariff: 200 units/month threshold\n"
        "RULES: Use realistic engineering estimates. Never invent numbers. "
        "Return ONLY valid JSON matching the schema. Battery is charged ONLY by "
        "solar PV. System is CAPTIVE - no export, surplus is curtailed."
    )

    user_prompt = """
SYSTEM
- Location: Lahore, Pakistan
- System: Captive solar + battery (NO export)
- Battery: solar-PV charged only
- LESCO limit: 200 kWh/month
- Solar: {solar_kwp} kWp | Battery: {battery_kwh} kWh

APPLIANCES
{table}

APPLIANCE MONTHLY ESTIMATE: {appl:.0f} kWh

12-MONTH HISTORY (Oct 2025 - Sep 2026)
{history}

TASKS
1. Estimate monthly solar PV generation for {solar_kwp} kWp in Lahore using
   realistic irradiance and ~20% system losses.
2. Provide SOLAR PV USAGE STRATEGY (3-5 concrete tips) for captive mode.
3. Provide FOUR TOP RECOMMENDATIONS:
   a. INCREASE SOLAR? (Yes/No + kWp)
   b. INCREASE BATTERY? (Yes/No + kWh)
   c. 2nd LESCO METER? (Recommended/Not recommended/Consider + legal warning)
   d. LOAD OPTIMIZATION (3-5 specific actions)
4. SHORT SUMMARY (2-3 sentences).

Return ONLY this JSON:
{{
  "solar_estimate": {{
    "daily_kwh": <number>, "monthly_kwh": <number>,
    "annual_kwh": <number>, "assumptions": "<short>"
  }},
  "solar_usage_strategy": ["tip1", "tip2", "tip3"],
  "recommendations": {{
    "increase_solar": {{"decision": "Yes"|"No", "amount_kwp": <num>, "reason": "<short>"}},
    "increase_battery": {{"decision": "Yes"|"No", "amount_kwh": <num>, "reason": "<short>"}},
    "second_meter": {{"decision": "Recommended"|"Not recommended"|"Consider", "reason": "<short>"}},
    "load_optimization": ["a1", "a2", "a3"]
  }},
  "summary": "<2-3 sentences>"
}}
""".format(solar_kwp=solar_kwp, battery_kwh=battery_kwh,
           table=appliance_table, appl=appliance_monthly, history=history_table)

    try:
        r = requests.post(
            GROQ_API_URL,
            headers={"Authorization": "Bearer " + api_key,
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}],
                  "temperature": 0.4, "max_tokens": 2500,
                  "response_format": {"type": "json_object"}},
            timeout=90,
        )
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code == 401:
            return {"error": "Invalid API Key. Check GROQ_API_KEY."}
        if code == 429:
            return {"error": "Rate limit reached. Wait 60 seconds."}
        return {"error": "HTTP {}: {}".format(code, str(e)[:200])}
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON from LLM: {}".format(str(e)[:200])}
    except Exception as e:
        return {"error": "AI call failed: {}".format(str(e)[:200])}


# =============================================================
# PAGE
# =============================================================
st.set_page_config(page_title="Home Energy Manager", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# -------------------------------------------------------------
# HIGH-CONTRAST CSS (identical to v3.1)
# -------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
        color: #F0F6FC !important;
    }
    html, body, [class*="css"] { color: #F0F6FC !important; }
    p, span, div, label, li, td, th { color: #F0F6FC !important; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; font-weight: 700 !important; }

    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #58A6FF, #3FB950, #FFDF4A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }

    section[data-testid="stSidebar"] {
        background: #0D1117 !important;
        border-right: 1px solid #30363D;
    }
    section[data-testid="stSidebar"] * { color: #F0F6FC !important; }

    div[data-testid="stMetric"] {
        background: #161B22;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #30363D;
    }
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label {
        color: #B0BAC5 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] {
        color: #58A6FF !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"] > div,
    div[data-testid="stMetricDelta"] {
        color: #3FB950 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    .info-card {
        background: #161B22;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        border-left: 4px solid #58A6FF;
        margin: 0.5rem 0;
        color: #F0F6FC !important;
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .info-card b { color: #FFFFFF !important; }

    .rec-card {
        background: #161B22;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 0.7rem 0;
        border-left: 5px solid #58A6FF;
        color: #F0F6FC !important;
    }
    .rec-danger  { border-left-color: #F85149; }
    .rec-warning { border-left-color: #D29922; }
    .rec-success { border-left-color: #3FB950; }
    .rec-info    { border-left-color: #58A6FF; }

    .rec-focus {
        font-size: 0.74rem; font-weight: 700;
        color: #58A6FF !important;
        letter-spacing: 1.2px; text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .rec-title {
        font-weight: 800; font-size: 1.12rem;
        color: #FFFFFF !important;
        margin-bottom: 0.4rem;
    }
    .rec-msg {
        font-size: 0.95rem;
        color: #C9D1D9 !important;
        line-height: 1.55;
    }

    .solar-card {
        background: linear-gradient(135deg, rgba(255,223,74,0.15), rgba(255,223,74,0.04));
        border: 1px solid rgba(255,223,74,0.55);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.5rem 0;
        color: #FFDF4A !important;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .solar-card b { color: #FFEB8A !important; }

    .captive-note {
        background: rgba(255,223,74,0.10);
        border: 1px solid rgba(255,223,74,0.45);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        color: #FFDF4A !important;
        margin: 0.5rem 0;
    }

    div[data-testid="stAlert"] { border-radius: 10px; padding: 0.9rem 1.1rem; }
    div[data-testid="stAlert"] * { color: inherit !important; }
    .stAlert p { color: #F0F6FC !important; }
    div[data-baseweb="notification"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
    }
    div[data-baseweb="notification"] * { color: #F0F6FC !important; }

    .stButton > button {
        background: linear-gradient(90deg, #1F6FEB, #58A6FF);
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.6rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #58A6FF, #1F6FEB);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88,166,255,0.4);
    }
    .stButton > button * { color: #FFFFFF !important; }

    input, textarea, select {
        background-color: #0D1117 !important;
        color: #F0F6FC !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    input:focus, textarea:focus, select:focus {
        border-color: #58A6FF !important;
        box-shadow: 0 0 0 2px rgba(88,166,255,0.25) !important;
    }

    div[data-testid="stExpander"] {
        background: #0D1117;
        border: 1px solid #30363D;
        border-radius: 10px;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: #F0F6FC !important;
        font-weight: 600;
    }
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] label { color: #F0F6FC !important; }

    div[data-testid="stDataFrame"] {
        background: #0D1117;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
    div[data-testid="stDataFrame"] * { color: #F0F6FC !important; }
    div[data-testid="stDataFrame"] th {
        background: #1F2937 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDataFrame"] td {
        background: #0D1117 !important;
        color: #F0F6FC !important;
    }

    div[data-testid="stSlider"] * { color: #F0F6FC !important; }
    div[data-testid="stSlider"] label { color: #F0F6FC !important; font-weight: 600; }
    div[data-testid="stCaptionContainer"] * { color: #A0AAB5 !important; }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span {
        color: #F0F6FC !important;
        line-height: 1.6;
    }
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] b {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    div[data-testid="stMarkdownContainer"] code {
        background: #21262D !important;
        color: #FFDF4A !important;
        padding: 2px 6px; border-radius: 4px;
    }

    div[data-testid="stSpinner"] * { color: #F0F6FC !important; }
    hr { border-color: #30363D !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# SESSION STATE
# =============================================================
if "appliances" not in st.session_state:
    st.session_state.appliances = [dict(a) for a in DEFAULT_APPLIANCES]
if "solar_kwp" not in st.session_state:
    st.session_state.solar_kwp = DEFAULT_SOLAR_KWP
if "battery_kwh" not in st.session_state:
    st.session_state.battery_kwh = DEFAULT_BATTERY_KWH
if "hist_df" not in st.session_state:
    st.session_state.hist_df = pd.DataFrame({
        "Month": MONTH_SEQ,
        "Consumption (kWh)": [None] * len(MONTH_SEQ),
        "Solar Gen (kWh)":   [None] * len(MONTH_SEQ),
        "Solar Used (kWh)":  [None] * len(MONTH_SEQ),
        "LESCO Import (kWh)":[None] * len(MONTH_SEQ),
        "Bill (PKR)":        [None] * len(MONTH_SEQ),
    })
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## 🔌 Appliances")
    st.caption("All fields editable. Days/week = 7 for daily use.")

    with st.expander("➕ Add Appliance", expanded=False):
        n = st.text_input("Name", key="nn")
        c1, c2, c3, c4 = st.columns(4)
        w = c1.number_input("W", min_value=1, max_value=20000, value=100, step=10, key="nw")
        h = c2.number_input("Hrs", min_value=0.1, max_value=24.0, value=1.0, step=0.25, key="nh")
        d = c3.number_input("Days/wk", min_value=1, max_value=7, value=7, step=1, key="nd")
        q = c4.number_input("Qty", min_value=1, max_value=50, value=1, step=1, key="nq")
        if st.button("Add", use_container_width=True) and n:
            st.session_state.appliances.append({
                "name": n, "watts": int(w), "hours": float(h),
                "days_per_week": int(d), "qty": int(q),
            })
            st.rerun()

    for i, app in enumerate(st.session_state.appliances):
        with st.expander("{} — {}W".format(app["name"][:26], app["watts"]), expanded=False):
            app["name"] = st.text_input("Name", value=app["name"], key="an_{}".format(i))
            cc = st.columns(2)
            app["watts"] = int(cc[0].number_input("Watts", min_value=1, max_value=20000,
                                                   value=int(app["watts"]), step=10, key="aw_{}".format(i)))
            app["hours"] = float(cc[1].number_input("Hrs/session", min_value=0.1, max_value=24.0,
                                                     value=float(app["hours"]), step=0.25, key="ah_{}".format(i)))
            app["days_per_week"] = int(cc[0].number_input("Days/week", min_value=1, max_value=7,
                                                          value=int(app["days_per_week"]), step=1, key="ad_{}".format(i)))
            app["qty"] = int(cc[1].number_input("Qty", min_value=1, max_value=50,
                                                value=int(app["qty"]), step=1, key="aq_{}".format(i)))
            if st.button("🗑️ Remove", key="d_{}".format(i)):
                st.session_state.appliances.pop(i)
                st.rerun()

    st.markdown("---")
    st.markdown("## ☀️ Solar & Battery")
    st.markdown(
        '<div class="captive-note">🔒 Captive system — no export. '
        'Battery charged only from solar PV.</div>',
        unsafe_allow_html=True,
    )
    st.session_state.solar_kwp = st.slider(
        "Solar PV (kWp)", 0.0, 30.0, st.session_state.solar_kwp, 0.5)
    st.session_state.battery_kwh = st.slider(
        "Battery (kWh)", 0.0, 80.0, st.session_state.battery_kwh, 1.0)

    st.markdown("---")
    st.markdown("## 📊 Monthly History")
    st.caption("Oct 2025 → Sep 2026. Fill only what you have.")
    st.session_state.hist_df = st.data_editor(
        st.session_state.hist_df,
        hide_index=True, use_container_width=True,
        num_rows="fixed", key="hist_editor",
    )

# =============================================================
# BUILD
# =============================================================
appliances = [Appliance(**a) for a in st.session_state.appliances]
appliance_monthly = sum(a.monthly_kwh for a in appliances)

history_df = st.session_state.hist_df
cons_list = _clean_col(history_df, "Consumption (kWh)")
bill_list = _clean_col(history_df, "Bill (PKR)")
avg_cons = sum(cons_list) / len(cons_list) if cons_list else appliance_monthly
avg_bill = sum(bill_list) / len(bill_list) if bill_list else 0

# =============================================================
# HEADER
# =============================================================
st.markdown('<div class="main-header">Home Energy Manager</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#B0BAC5;'>"
    "September 2026 | LESCO Protected Consumer | "
    "Captive Solar (No Export) | LLM-Driven Analysis</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =============================================================
# METRICS
# =============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Appliance Est.", "{:.0f} kWh/mo".format(appliance_monthly))
with c2:
    st.metric("Metered Avg (12-mo)", "{:.0f} kWh/mo".format(avg_cons))
with c3:
    st.metric("Avg Bill", "PKR {:,.0f}".format(avg_bill) if avg_bill else "—")
with c4:
    gap = appliance_monthly - avg_cons
    st.metric("Est. vs Metered", "{:+.0f} kWh".format(gap))

# =============================================================
# SYSTEM SUMMARY
# =============================================================
st.markdown("### ⚙️ System Configuration")
sc1, sc2 = st.columns(2)
with sc1:
    st.markdown(
        '<div class="solar-card">'
        '<b>Solar PV:</b> {:.1f} kWp<br>'
        '<b>Battery:</b> {:.1f} kWh<br>'
        '<b>Charging:</b> Solar PV only<br>'
        '<b>Export:</b> Disabled (captive)'
        '</div>'.format(st.session_state.solar_kwp, st.session_state.battery_kwh),
        unsafe_allow_html=True,
    )
with sc2:
    st.markdown(
        '<div class="info-card">'
        '<b>Protected limit:</b> 200 kWh/month<br>'
        '<b>Months of history:</b> {}<br>'
        '<b>Appliances tracked:</b> {}<br>'
        '<b>Engine:</b> Groq · openai/gpt-oss-120b'
        '</div>'.format(len(cons_list), len(appliances)),
        unsafe_allow_html=True,
    )

# =============================================================
# MODERN CHARTS ROW 1 — Consumption trend + Donut
# =============================================================
st.markdown("### 📊 Consumption Analytics")
col_left, col_right = st.columns([1.15, 1])

with col_left:
    fig_line = chart_consumption_line(history_df, appliance_monthly)
    if fig_line:
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("📝 Add monthly Consumption (kWh) in the sidebar to see the trend.")

with col_right:
    app_df = pd.DataFrame([{
        "Appliance": a.name,
        "Watts": a.watts,
        "Hrs/Session": a.hours,
        "Days/Week": a.days_per_week,
        "Qty": a.qty,
        "kWh/Month": round(a.monthly_kwh, 1),
    } for a in appliances])
    st.plotly_chart(chart_appliance_modern(app_df), use_container_width=True)

# =============================================================
# MODERN CHARTS ROW 2 — Stacked + Bill Trend
# =============================================================
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.plotly_chart(chart_monthly_stacked(history_df), use_container_width=True)

with col_r2:
    fig_bill = chart_bill_trend(history_df)
    if fig_bill:
        st.plotly_chart(fig_bill, use_container_width=True)
    else:
        st.info("📝 Add monthly Bill (PKR) in the sidebar to see the bill trend.")

# =============================================================
# SOLAR UTILIZATION (if data present)
# =============================================================
fig_util = chart_solar_utilization(history_df)
if fig_util:
    st.plotly_chart(fig_util, use_container_width=True)

# =============================================================
# APPLIANCE TABLE
# =============================================================
st.markdown("### 🔌 Appliance Detail")
app_detail = app_df.copy()
app_detail["% of Total"] = (app_detail["kWh/Month"] / appliance_monthly * 100).round(1) if appliance_monthly > 0 else 0
st.dataframe(app_detail, hide_index=True, use_container_width=True)

# =============================================================
# HISTORY TABLE VIEW
# =============================================================
st.markdown("### 📊 12-Month Data (Oct 2025 → Sep 2026)")
st.dataframe(st.session_state.hist_df, hide_index=True, use_container_width=True)

# =============================================================
# LLM ANALYSIS
# =============================================================
st.markdown("---")
st.markdown("### 🤖 LLM Analysis (Groq · gpt-oss-120b)")
st.caption(
    "The LLM estimates solar PV generation from real Lahore irradiance and "
    "system loss factors, then produces strategy + 4 focus recommendations."
)

if st.button("🚀 Run LLM Analysis", use_container_width=True):
    with st.spinner("Analyzing with Groq..."):
        st.session_state.analysis = call_llm_analysis(
            solar_kwp=st.session_state.solar_kwp,
            battery_kwh=st.session_state.battery_kwh,
            appliances_df=appliances,
            history_df=history_df,
        )

analysis = st.session_state.analysis

if analysis is None:
    st.info("Click **Run LLM Analysis** to generate results.")
elif "error" in analysis:
    st.error("⚠️ " + analysis["error"])
else:
    # ---------- SOLAR ESTIMATE ----------
    se = analysis.get("solar_estimate", {})
    if se:
        st.markdown("#### ☀️ Solar PV Generation Estimate")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Daily", "{:.1f} kWh".format(se.get("daily_kwh", 0)))
        with e2:
            st.metric("Monthly", "{:.0f} kWh".format(se.get("monthly_kwh", 0)))
        with e3:
            st.metric("Annual", "{:,.0f} kWh".format(se.get("annual_kwh", 0)))
        if se.get("assumptions"):
            st.markdown(
                '<div class="info-card">'
                '<b>Assumptions:</b> {}'
                '</div>'.format(se["assumptions"]),
                unsafe_allow_html=True,
            )

    # ---------- STRATEGY ----------
    strategy = analysis.get("solar_usage_strategy", [])
    if strategy:
        st.markdown("#### 💡 Solar PV Usage Strategy")
        for i, tip in enumerate(strategy, 1):
            st.markdown(
                '<div class="info-card">'
                '<b>{}. </b>{}'
                '</div>'.format(i, tip),
                unsafe_allow_html=True,
            )

    # ---------- RECOMMENDATIONS ----------
    recs = analysis.get("recommendations", {})
    if recs:
        st.markdown("#### 🎯 Top Recommendations")

        inc_solar = recs.get("increase_solar", {})
        if inc_solar:
            dec = inc_solar.get("decision", "?")
            level = "rec-success" if dec == "No" else "rec-warning"
            amount = inc_solar.get("amount_kwp", 0) or 0
            title = "No — Solar is adequate" if dec == "No" else \
                    "Yes — Add ~{:.1f} kWp".format(amount)
            st.markdown(
                '<div class="rec-card {}">'
                '<div class="rec-focus">◆ Increase Solar?</div>'
                '<div class="rec-title">{}</div>'
                '<div class="rec-msg">{}</div>'
                '</div>'.format(level, title, inc_solar.get("reason", "")),
                unsafe_allow_html=True,
            )

        inc_batt = recs.get("increase_battery", {})
        if inc_batt:
            dec = inc_batt.get("decision", "?")
            level = "rec-success" if dec == "No" else "rec-warning"
            amount = inc_batt.get("amount_kwh", 0) or 0
            title = "No — Battery is adequate" if dec == "No" else \
                    "Yes — Add ~{:.1f} kWh".format(amount)
            st.markdown(
                '<div class="rec-card {}">'
                '<div class="rec-focus">◆ Increase Battery?</div>'
                '<div class="rec-title">{}</div>'
                '<div class="rec-msg">{}</div>'
                '</div>'.format(level, title, inc_batt.get("reason", "")),
                unsafe_allow_html=True,
            )

        meter = recs.get("second_meter", {})
        if meter:
            dec = meter.get("decision", "?")
            if "Not" in dec:
                level = "rec-danger"
            elif "Consider" in dec:
                level = "rec-warning"
            else:
                level = "rec-info"
            st.markdown(
                '<div class="rec-card {}">'
                '<div class="rec-focus">◆ 2nd LESCO Meter?</div>'
                '<div class="rec-title">{}</div>'
                '<div class="rec-msg">{}</div>'
                '</div>'.format(level, dec, meter.get("reason", "")),
                unsafe_allow_html=True,
            )

        opts = recs.get("load_optimization", [])
        if opts:
            st.markdown(
                '<div class="rec-card rec-info">'
                '<div class="rec-focus">◆ Load Optimization</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            for i, o in enumerate(opts, 1):
                st.markdown(
                    '<div class="info-card"><b>{}. </b>{}</div>'.format(i, o),
                    unsafe_allow_html=True,
                )

    # ---------- SUMMARY ----------
    summary = analysis.get("summary")
    if summary:
        st.markdown("#### 📝 Summary")
        st.success(summary)

# =============================================================
# LEGAL NOTICE
# =============================================================
st.markdown("---")
st.warning(
    "**Legal Notice (September 2026):** LESCO is actively cracking down on "
    "multiple meters used to artificially stay below 200 units. Households "
    "found doing this immediately lose protected status. Multiple meters are "
    "only permissible where **separate families occupy distinct portions with "
    "independent kitchens and wiring**."
)

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#A0AAB5; font-size:0.8rem;">'
    'Home Energy Manager v3.2 | Oct 2025 → Sep 2026 | '
    'Captive Solar (No Export) | Groq · openai/gpt-oss-120b'
    '</p>',
    unsafe_allow_html=True,
)
