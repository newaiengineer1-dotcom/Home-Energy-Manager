"""
Home Energy Management Dashboard
LESCO Protected Consumer Optimizer - September 2026

Captive solar (no export, no net metering). Battery solar-charged only.
Solar generation ESTIMATED from kWp (not hallucinated).
AI provides solar-usage strategy + 4 focus recommendations.
"""
import os
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dataclasses import dataclass, field

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
DEFAULT_PROTECTED_LIMIT = 200.0
DEFAULT_PROTECTED_RATE = 13.40
DEFAULT_UNPROTECTED_RATE = 42.80
DEFAULT_SOLAR_KWP = 5.0
DEFAULT_BATTERY_KWH = 10.0

LAHORE_PEAK_SUN_HOURS = 5.2      # kWh/m2/day annual avg for Lahore
DAY_LOAD_RATIO = 0.55            # 55% of daily load is during solar hours
BATTERY_EFFICIENCY = 0.85        # round-trip LiFePO4
SYSTEM_LOSS_FACTOR = 0.80        # inverter + wiring + soiling losses

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# 14-month window: Aug 2025 -> Sep 2026
DEFAULT_MONTH_SEQ = [
    "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25",
    "Jan-26", "Feb-26", "Mar-26", "Apr-26", "May-26",
    "Jun-26", "Jul-26", "Aug-26", "Sep-26",
]

COLORS = {
    "bg": "#0D1117", "bg2": "#161B22", "border": "#30363D",
    "text": "#C9D1D9", "muted": "#8B949E",
    "primary": "#58A6FF", "success": "#3FB950",
    "warning": "#D29922", "danger": "#F85149",
    "solar": "#FFDF4A", "battery": "#39D353", "grid": "#58A6FF",
}

# =============================================================
# DEFAULT APPLIANCES - Your setup
# =============================================================
DEFAULT_APPLIANCES = [
    {"name": "Inverter AC 1.5T @26C (2x/week, 6h)",  "watts": 550,  "hours": 6.0,  "days_per_week": 2, "qty": 1},
    {"name": "Inverter AC 1.5T @26C (daily, 16h)",   "watts": 550,  "hours": 16.0, "days_per_week": 7, "qty": 1},
    {"name": "Inverter AC 1.0T @26C (daily, 16h)",   "watts": 420,  "hours": 16.0, "days_per_week": 7, "qty": 1},
    {"name": "Microwave Oven (15 min/day)",          "watts": 1200, "hours": 0.25, "days_per_week": 7, "qty": 1},
    {"name": "Washing Machine (4h/week)",            "watts": 500,  "hours": 4.0,  "days_per_week": 1, "qty": 1},
    {"name": "Submersible Pump (2h/day)",            "watts": 750,  "hours": 2.0,  "days_per_week": 7, "qty": 1},
    {"name": "Machine 3kW (5h, 2x/week)",            "watts": 3000, "hours": 5.0,  "days_per_week": 2, "qty": 1},
    {"name": "RO Plant 1.5kW (5h, 2x/week)",         "watts": 1500, "hours": 5.0,  "days_per_week": 2, "qty": 1},
    {"name": "Ceiling Fan (BLDC)",                   "watts": 28,   "hours": 12.0, "days_per_week": 7, "qty": 6},
    {"name": "LED Light 12W",                        "watts": 12,   "hours": 6.0,  "days_per_week": 7, "qty": 30},
    {"name": "Philips Iron (3h/week)",               "watts": 1000, "hours": 3.0,  "days_per_week": 1, "qty": 1},
]

# =============================================================
# DATA MODELS
# =============================================================
@dataclass
class Appliance:
    name: str
    watts: float
    hours: float
    days_per_week: int = 7
    qty: int = 1

    @property
    def kwh_per_active_day(self) -> float:
        return (self.watts * self.hours * self.qty) / 1000

    @property
    def weekly_kwh(self) -> float:
        return self.kwh_per_active_day * self.days_per_week

    @property
    def daily_avg_kwh(self) -> float:
        return self.weekly_kwh / 7.0

    @property
    def monthly_kwh(self) -> float:
        return self.weekly_kwh * 4.33


@dataclass
class SystemConfig:
    solar_kwp: float = DEFAULT_SOLAR_KWP
    battery_kwh: float = DEFAULT_BATTERY_KWH


@dataclass
class Tariff:
    protected_limit: float = DEFAULT_PROTECTED_LIMIT
    protected_rate: float = DEFAULT_PROTECTED_RATE
    unprotected_rate: float = DEFAULT_UNPROTECTED_RATE
    calibrated: bool = False
    source: str = "defaults"

    def bill_for(self, units: float) -> dict:
        if units <= 0:
            return {"total": 0, "per_unit": 0, "p_units": 0,
                    "u_units": 0, "p_cost": 0, "u_cost": 0}
        p_units = min(units, self.protected_limit)
        u_units = max(0.0, units - self.protected_limit)
        p_cost = p_units * self.protected_rate
        u_cost = u_units * self.unprotected_rate
        total = p_cost + u_cost
        return {
            "total": round(total),
            "per_unit": round(total / units, 2) if units > 0 else 0,
            "p_units": round(p_units), "u_units": round(u_units),
            "p_cost": round(p_cost), "u_cost": round(u_cost),
        }


@dataclass
class Snapshot:
    appliances: list
    system: SystemConfig
    tariff: Tariff
    total_daily_kwh: float = 0.0
    total_monthly_kwh: float = 0.0
    day_load_daily: float = 0.0
    night_load_daily: float = 0.0
    solar_daily_kwh: float = 0.0
    solar_monthly_kwh: float = 0.0
    solar_direct_monthly: float = 0.0
    solar_to_battery_monthly: float = 0.0
    solar_curtailed_monthly: float = 0.0
    battery_discharge_monthly: float = 0.0
    grid_import: float = 0.0
    is_protected: bool = False
    remaining: float = 0.0
    bill: dict = field(default_factory=dict)


# =============================================================
# SOLAR GENERATION ESTIMATOR (formula-based, NOT hallucinated)
# =============================================================
def estimate_solar_generation(kwp: float) -> dict:
    """
    Realistic solar generation estimate for Lahore, Pakistan.
    Formula: kWp x Peak-Sun-Hours x Days x System-Loss-Factor
    """
    daily_gross = kwp * LAHORE_PEAK_SUN_HOURS
    daily_net = daily_gross * SYSTEM_LOSS_FACTOR
    monthly = daily_net * 30
    return {
        "daily_gross": round(daily_gross, 2),
        "daily_net": round(daily_net, 2),
        "monthly_net": round(monthly, 1),
    }


# =============================================================
# TARIFF CALIBRATION
# =============================================================
def _parse_csv(text: str) -> list:
    if not text:
        return []
    tokens = text.replace("\n", ",").replace(" ", ",").split(",")
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def calibrate_tariff(kwh_csv: str, bill_csv: str) -> Tariff:
    t = Tariff()
    kwh = _parse_csv(kwh_csv)
    bills = _parse_csv(bill_csv)
    if len(kwh) >= 3 and len(bills) >= 3 and len(kwh) == len(bills):
        rates = [(b / k) if k > 0 else 0.0 for k, b in zip(kwh, bills)]
        pairs = sorted(zip(kwh, rates))
        max_jump, threshold = 0.0, DEFAULT_PROTECTED_LIMIT
        for i in range(1, len(pairs)):
            u_prev, r_prev = pairs[i - 1]
            u_curr, r_curr = pairs[i]
            if 100 <= u_prev and u_curr <= 400:
                jump = r_curr - r_prev
                if jump > max_jump:
                    max_jump = jump
                    threshold = (u_prev + u_curr) / 2.0
        protected = [r for u, r in pairs if u <= threshold and r > 0]
        unprotected = [r for u, r in pairs if u > threshold and r > 0]
        if protected:
            t.protected_rate = round(sum(protected) / len(protected), 2)
        if unprotected:
            t.unprotected_rate = round(sum(unprotected) / len(unprotected), 2)
        t.protected_limit = round(threshold)
        t.calibrated = True
        t.source = "calibrated from {} months".format(len(kwh))
    elif len(kwh) >= 3:
        t.source = "defaults (kWh history: {} months)".format(len(kwh))
    elif len(bills) >= 3:
        avg_bill = sum(bills) / len(bills)
        t.source = "defaults (avg bill: PKR {:,.0f})".format(avg_bill)
    return t


# =============================================================
# ENERGY COMPUTATION (captive solar)
# =============================================================
def build_snapshot(appliances, system, tariff):
    s = Snapshot(appliances=appliances, system=system, tariff=tariff)
    s.total_daily_kwh = sum(a.daily_avg_kwh for a in appliances)
    s.total_monthly_kwh = s.total_daily_kwh * 30

    s.day_load_daily = s.total_daily_kwh * DAY_LOAD_RATIO
    s.night_load_daily = s.total_daily_kwh * (1.0 - DAY_LOAD_RATIO)

    # Solar generation via real formula (loss-adjusted)
    solar_est = estimate_solar_generation(system.solar_kwp)
    s.solar_daily_kwh = solar_est["daily_net"]
    s.solar_monthly_kwh = solar_est["monthly_net"]

    # 1. Direct solar to day-load
    solar_direct = min(s.solar_daily_kwh, s.day_load_daily)
    # 2. Surplus -> battery (solar only, never grid)
    solar_surplus = max(0.0, s.solar_daily_kwh - solar_direct)
    battery_charge_in = min(solar_surplus, system.battery_kwh)
    # 3. Remaining surplus is curtailed (no export)
    solar_curtailed = solar_surplus - battery_charge_in
    # 4. Battery discharge at night
    battery_usable = battery_charge_in * BATTERY_EFFICIENCY
    battery_discharge = min(battery_usable, s.night_load_daily)
    # 5. Grid covers the rest
    grid_daily = max(0.0, s.total_daily_kwh - solar_direct - battery_discharge)

    s.solar_direct_monthly = solar_direct * 30
    s.solar_to_battery_monthly = battery_charge_in * 30
    s.solar_curtailed_monthly = solar_curtailed * 30
    s.battery_discharge_monthly = battery_discharge * 30
    s.grid_import = grid_daily * 30

    s.is_protected = s.grid_import <= tariff.protected_limit
    s.remaining = tariff.protected_limit - s.grid_import
    s.bill = tariff.bill_for(s.grid_import)
    return s


# =============================================================
# RECOMMENDATIONS ENGINE (4 focus areas)
# =============================================================
def generate_recommendations(snap, hist):
    recs = []
    avg_lesco = hist["avg_lesco"]
    avg_solar_gen = hist["avg_solar_gen"]
    avg_solar_used = hist["avg_solar_used"]

    # FOCUS 1: Increase Solar?
    if avg_lesco > 200:
        shortfall = avg_lesco - 200
        extra_kwp = shortfall / (LAHORE_PEAK_SUN_HOURS * SYSTEM_LOSS_FACTOR * 30)
        recs.append({
            "focus": "INCREASE SOLAR?",
            "level": "danger",
            "title": "Yes — undersized by ~{:.1f} kWp".format(extra_kwp),
            "message": (
                "Metered LESCO import averages {:.0f} units/month. To bring it "
                "under 200, add ~{:.1f} kWp solar (accounting for 80% system "
                "efficiency).".format(avg_lesco, extra_kwp)
            ),
            "action": "Add {:.1f} kWp solar".format(extra_kwp),
        })
    elif avg_lesco > 180:
        recs.append({
            "focus": "INCREASE SOLAR?",
            "level": "warning",
            "title": "Marginal — near the 200-unit limit",
            "message": "Average LESCO import {:.0f} units. Add 1-2 kWp for buffer.".format(avg_lesco),
            "action": "Add 1-2 kWp for safety margin",
        })
    else:
        recs.append({
            "focus": "INCREASE SOLAR?",
            "level": "success",
            "title": "No — Solar is adequate",
            "message": "Average LESCO import {:.0f} units, under 200.".format(avg_lesco),
            "action": "Maintain current solar capacity",
        })

    # FOCUS 2: Increase Battery?
    ideal_battery = snap.night_load_daily / BATTERY_EFFICIENCY
    util_pct = (avg_solar_used / avg_solar_gen * 100.0) if avg_solar_gen > 0 else 100.0
    wasted = max(0.0, avg_solar_gen - avg_solar_used)

    if snap.system.battery_kwh < ideal_battery * 0.8:
        recs.append({
            "focus": "INCREASE BATTERY?",
            "level": "warning",
            "title": "Yes — Battery is undersized",
            "message": (
                "Night load ~{:.1f} kWh/night. Ideal battery ~{:.1f} kWh; "
                "you have {:.1f} kWh.".format(
                    snap.night_load_daily, ideal_battery, snap.system.battery_kwh)
            ),
            "action": "Increase battery to ~{:.0f} kWh".format(ideal_battery),
        })
    elif util_pct < 70 and wasted > 20:
        recs.append({
            "focus": "INCREASE BATTERY?",
            "level": "warning",
            "title": "Yes — Utilization only {:.0f}%".format(util_pct),
            "message": (
                "~{:.0f} units/month curtailed. No export, so larger battery "
                "captures this surplus.".format(wasted)
            ),
            "action": "Increase battery to ~{:.0f} kWh".format(ideal_battery),
        })
    elif snap.system.battery_kwh > ideal_battery * 1.5:
        recs.append({
            "focus": "INCREASE BATTERY?",
            "level": "info",
            "title": "No — Battery may be oversized",
            "message": (
                "Battery {:.0f} kWh >> night load {:.1f} kWh.".format(
                    snap.system.battery_kwh, snap.night_load_daily)
            ),
            "action": "Current battery is sufficient",
        })
    else:
        recs.append({
            "focus": "INCREASE BATTERY?",
            "level": "success",
            "title": "No — Battery is well-sized",
            "message": "Battery capacity matches night load profile.",
            "action": "Maintain current battery",
        })

    # FOCUS 3: 2nd LESCO Meter?
    if 200 < avg_lesco < 450:
        recs.append({
            "focus": "2nd LESCO METER?",
            "level": "danger",
            "title": "Not recommended — legal risk",
            "message": (
                "2nd meters only allowed for genuinely separate families with "
                "independent kitchens and wiring. LESCO actively cracks down."
            ),
            "action": "Prefer solar/battery expansion over 2nd meter",
        })
    elif avg_lesco >= 450:
        recs.append({
            "focus": "2nd LESCO METER?",
            "level": "warning",
            "title": "Consider only if legally separable",
            "message": (
                "Consumption is very high. If genuinely separable, a 2nd meter "
                "may be defensible — otherwise expand solar."
            ),
            "action": "Evaluate legal 2nd meter vs. more solar",
        })
    else:
        recs.append({
            "focus": "2nd LESCO METER?",
            "level": "info",
            "title": "Not needed",
            "message": "Within single protected-tier consumption.",
            "action": "No 2nd meter required",
        })

    # FOCUS 4: Load Optimization
    if snap.appliances:
        biggest = max(snap.appliances, key=lambda a: a.monthly_kwh)
        pct = (biggest.monthly_kwh / max(snap.total_monthly_kwh, 1)) * 100
        recs.append({
            "focus": "LOAD OPTIMIZATION",
            "level": "info",
            "title": "Biggest load: {} ({:.0f}%)".format(biggest.name, pct),
            "message": (
                "'{}' uses {:.0f} kWh/month. Shift to 10 AM–3 PM for direct "
                "solar use.".format(biggest.name, biggest.monthly_kwh)
            ),
            "action": "Shift {} to 10 AM – 3 PM".format(biggest.name),
        })
    if avg_solar_gen > 0 and snap.solar_monthly_kwh > 0:
        gap = snap.solar_monthly_kwh - avg_solar_gen
        if abs(gap) > max(50, snap.solar_monthly_kwh * 0.2):
            recs.append({
                "focus": "LOAD OPTIMIZATION",
                "level": "info",
                "title": "Solar estimate vs actual",
                "message": (
                    "Formula estimate = {:.0f} kWh/month, your metered = "
                    "{:.0f} kWh/month. Difference: {:.0f}.".format(
                        snap.solar_monthly_kwh, avg_solar_gen, gap)
                ),
                "action": "Reconcile if measurements are reliable",
            })
    if hist["avg_consumption"] > 0 and snap.total_monthly_kwh > 0:
        diff_pct = ((snap.total_monthly_kwh - hist["avg_consumption"]) /
                    hist["avg_consumption"]) * 100.0
        if abs(diff_pct) > 20:
            recs.append({
                "focus": "LOAD OPTIMIZATION",
                "level": "warning",
                "title": "Estimate off by {:.0f}%".format(abs(diff_pct)),
                "message": (
                    "Appliance estimate {:.0f} kWh, metered {:.0f} kWh."
                ).format(snap.total_monthly_kwh, hist["avg_consumption"]),
                "action": "Fine-tune appliance wattages",
            })

    return recs


# =============================================================
# AI — SOLAR USAGE STRATEGY (not hallucinating generation)
# =============================================================
def run_solar_strategy(snap, hist):
    """
    Ask the LLM for a SOLAR USAGE STRATEGY given the numbers we computed.
    We pass pre-computed generation figures, so the LLM does NOT hallucinate.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return "AI advisor disabled. Add GROQ_API_KEY to Streamlit Secrets."

    # Computed values — given to the LLM so it cannot hallucinate
    solar_monthly = snap.solar_monthly_kwh
    solar_direct = snap.solar_direct_monthly
    solar_batt = snap.solar_to_battery_monthly
    solar_curtail = snap.solar_curtailed_monthly
    grid_import = snap.grid_import
    total_load = snap.total_monthly_kwh
    battery_kwh = snap.system.battery_kwh
    night_load = snap.night_load_daily
    daytime_load = snap.day_load_daily

    appliance_summary = "; ".join(
        "{}: {}W × {}h × {}d/wk".format(a.name, a.watts, a.hours, a.days_per_week)
        for a in snap.appliances
    )

    sys_p = (
        "You are an expert energy advisor for a Lahore, Pakistan household. "
        "The system is CAPTIVE solar + battery — NO grid export, NO net metering. "
        "Surplus solar is CURTAILED. Battery is charged ONLY from solar, never grid. "
        "You will be given COMPUTED numbers — do not re-estimate them. "
        "Answer concisely in plain text, no markdown headers."
    )
    usr_p = (
        "COMPUTED SYSTEM STATE (do not recompute):\n"
        "- Total load: {total:.0f} kWh/month\n"
        "- Solar generation (formula): {solar:.0f} kWh/month\n"
        "- Solar used directly: {direct:.0f} kWh/month\n"
        "- Solar to battery: {batt:.0f} kWh/month\n"
        "- Solar curtailed (wasted): {curtail:.0f} kWh/month\n"
        "- Battery capacity: {bcap:.0f} kWh\n"
        "- Night load: {night:.1f} kWh/day\n"
        "- Day load: {day:.1f} kWh/day\n"
        "- Grid import: {grid:.0f} kWh/month\n"
        "- Appliances: {appl}\n"
        "\nProvide:\n"
        "1. SOLAR USAGE STRATEGY — 3 specific rules to maximize on-site "
        "consumption (e.g., schedule changes, load shifting).\n"
        "2. BATTERY DISPATCH STRATEGY — how to use the battery daily.\n"
        "3. IF curtailed is >10% of generation, what should be done?\n"
        "4. IF grid import >180, which single action helps most?\n"
    ).format(
        total=total_load, solar=solar_monthly, direct=solar_direct,
        batt=solar_batt, curtail=solar_curtail, bcap=battery_kwh,
        night=night_load, day=daytime_load,
        grid=grid_import, appl=appliance_summary,
    )

    try:
        r = requests.post(
            GROQ_API_URL,
            headers={"Authorization": "Bearer " + api_key,
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "system", "content": sys_p},
                               {"role": "user", "content": usr_p}],
                  "temperature": 0.6, "max_tokens": 1400},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code == 401:
            return "Invalid API Key. Check GROQ_API_KEY in Streamlit Secrets."
        if code == 429:
            return "Rate limit reached. Wait 60 seconds and try again."
        return "AI failed: HTTP {}".format(code)
    except Exception as e:
        return "AI failed: {}".format(e)


# =============================================================
# PAGE CONFIG + CSS
# =============================================================
st.set_page_config(page_title="Home Energy Manager", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0D1117 0%, #161B22 100%); }
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #58A6FF, #3FB950, #FFDF4A);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: #161B22; border-radius: 12px;
        padding: 1rem; border: 1px solid #30363D;
    }
    .tariff-badge {
        display: inline-block; padding: 0.35rem 0.9rem;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        background: rgba(88,166,255,0.15); color: #58A6FF;
        border: 1px solid rgba(88,166,255,0.35);
    }
    .tariff-badge-ok {
        background: rgba(63,185,80,0.15); color: #3FB950;
        border-color: rgba(63,185,80,0.35);
    }
    .rec-card {
        background: #161B22; border-left: 4px solid #58A6FF;
        border-radius: 10px; padding: 0.9rem 1.2rem;
        margin: 0.6rem 0; color: #C9D1D9;
    }
    .rec-danger  { border-left-color: #F85149; }
    .rec-warning { border-left-color: #D29922; }
    .rec-success { border-left-color: #3FB950; }
    .rec-info    { border-left-color: #58A6FF; }
    .rec-focus {
        font-size: 0.72rem; font-weight: 700;
        color: #58A6FF; letter-spacing: 1px;
        text-transform: uppercase; margin-bottom: 0.15rem;
    }
    .rec-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.25rem; }
    .rec-msg   { font-size: 0.9rem; color: #8B949E; margin-bottom: 0.4rem; }
    .rec-action{ font-size: 0.88rem; color: #3FB950; font-weight: 600; }
    .captive-note {
        background: rgba(255,223,74,0.08);
        border: 1px solid rgba(255,223,74,0.3);
        border-radius: 10px; padding: 0.6rem 0.9rem;
        font-size: 0.82rem; color: #FFDF4A; margin: 0.5rem 0;
    }
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
        "Month": DEFAULT_MONTH_SEQ,
        "Consumption (kWh)":   [None] * 14,
        "Solar Gen (kWh)":     [None] * 14,
        "Solar Used (kWh)":    [None] * 14,
        "LESCO Import (kWh)":  [None] * 14,
        "Bill (PKR)":          [None] * 14,
    })
if "ai_result" not in st.session_state:
    st.session_state.ai_result = ""

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## 🔌 Appliance Manager")
    st.caption("All fields editable. Days/week = 7 for daily appliances.")

    with st.expander("➕ Add New Appliance", expanded=False):
        n_name = st.text_input("Name", key="nn")
        c1, c2, c3, c4 = st.columns(4)
        n_w = c1.number_input("W", min_value=1, max_value=20000,
                              value=100, step=10, key="nw")
        n_h = c2.number_input("Hrs", min_value=0.1, max_value=24.0,
                              value=1.0, step=0.25, key="nh")
        n_d = c3.number_input("Days/wk", min_value=1, max_value=7,
                              value=7, step=1, key="nd")
        n_q = c4.number_input("Qty", min_value=1, max_value=50,
                              value=1, step=1, key="nq")
        if st.button("Add Appliance", use_container_width=True) and n_name:
            st.session_state.appliances.append({
                "name": n_name, "watts": int(n_w), "hours": float(n_h),
                "days_per_week": int(n_d), "qty": int(n_q),
            })
            st.rerun()

    st.markdown("### 📋 Current Appliances")
    for i, app in enumerate(st.session_state.appliances):
        label = "{} — {}W".format(app["name"][:28], app["watts"])
        with st.expander(label, expanded=False):
            app["name"] = st.text_input("Name", value=app["name"], key="an_{}".format(i))
            cc = st.columns(2)
            app["watts"] = int(cc[0].number_input(
                "Watts", min_value=1, max_value=20000,
                value=int(app["watts"]), step=10, key="aw_{}".format(i)))
            app["hours"] = float(cc[1].number_input(
                "Hrs/session", min_value=0.1, max_value=24.0,
                value=float(app["hours"]), step=0.25, key="ah_{}".format(i)))
            app["days_per_week"] = int(cc[0].number_input(
                "Days/week", min_value=1, max_value=7,
                value=int(app["days_per_week"]), step=1, key="ad_{}".format(i)))
            app["qty"] = int(cc[1].number_input(
                "Qty", min_value=1, max_value=50,
                value=int(app["qty"]), step=1, key="aq_{}".format(i)))
            if st.button("🗑️ Remove", key="d_{}".format(i)):
                st.session_state.appliances.pop(i)
                st.rerun()

    st.markdown("---")
    st.markdown("## ☀️ Solar & Battery")
    st.markdown(
        '<div class="captive-note">🔒 <b>Captive system</b> — no export. '
        'Battery charged only from solar.</div>',
        unsafe_allow_html=True,
    )
    st.session_state.solar_kwp = st.slider(
        "Solar PV System (kWp)", 0.0, 30.0, st.session_state.solar_kwp, 0.5)
    st.session_state.battery_kwh = st.slider(
        "Battery Capacity (kWh)", 0.0, 80.0, st.session_state.battery_kwh, 1.0)

    # Live solar estimate display
    _est = estimate_solar_generation(st.session_state.solar_kwp)
    st.info(
        "**Solar estimate (formula):**\n"
        "- Daily (net): {:.1f} kWh\n"
        "- Monthly: {:.0f} kWh".format(
            _est["daily_net"], _est["monthly_net"])
    )

    st.markdown("---")
    st.markdown("## 📊 Monthly History")
    st.caption("Aug 2025 → Sep 2026. Fill only what you know.")
    st.session_state.hist_df = st.data_editor(
        st.session_state.hist_df,
        hide_index=True, use_container_width=True,
        num_rows="fixed", key="hist_editor",
    )

# =============================================================
# BUILD SNAPSHOT
# =============================================================
appliances = [Appliance(**a) for a in st.session_state.appliances]
system = SystemConfig(solar_kwp=st.session_state.solar_kwp,
                      battery_kwh=st.session_state.battery_kwh)

df = st.session_state.hist_df.copy()
def _clean(series):
    return pd.to_numeric(series, errors="coerce").dropna().tolist()

hist_lesco = _clean(df["LESCO Import (kWh)"])
hist_bills = _clean(df["Bill (PKR)"])
hist_consumption = _clean(df["Consumption (kWh)"])
hist_solar_gen = _clean(df["Solar Gen (kWh)"])
hist_solar_used = _clean(df["Solar Used (kWh)"])

tariff = calibrate_tariff(",".join(map(str, hist_lesco)),
                          ",".join(map(str, hist_bills)))
snap = build_snapshot(appliances, system, tariff)

# History averages (fall back to computed values)
hist = {
    "avg_consumption": sum(hist_consumption)/len(hist_consumption) if hist_consumption else snap.total_monthly_kwh,
    "avg_solar_gen": sum(hist_solar_gen)/len(hist_solar_gen) if hist_solar_gen else snap.solar_monthly_kwh,
    "avg_solar_used": sum(hist_solar_used)/len(hist_solar_used) if hist_solar_used else snap.solar_direct_monthly,
    "avg_lesco": sum(hist_lesco)/len(hist_lesco) if hist_lesco else snap.grid_import,
    "avg_bill": sum(hist_bills)/len(hist_bills) if hist_bills else snap.bill["total"],
}

# =============================================================
# HEADER
# =============================================================
st.markdown('<div class="main-header">Home Energy Management Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#8B949E;'>"
    "September 2026 | LESCO Protected Consumer | Captive Solar (No Export)</p>",
    unsafe_allow_html=True,
)

badge_class = "tariff-badge tariff-badge-ok" if tariff.calibrated else "tariff-badge"
badge_text = ("OK " if tariff.calibrated else "DEFAULTS ") + "Tariff: " + tariff.source \
    + " | Protected <=" + str(int(tariff.protected_limit)) \
    + " @ PKR " + str(tariff.protected_rate) + "/unit" \
    + " | Unprotected @ PKR " + str(tariff.unprotected_rate) + "/unit"
st.markdown(
    "<div style='text-align:center; margin: 0.5rem 0 1rem 0;'>"
    "<span class='" + badge_class + "'>" + badge_text + "</span></div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =============================================================
# KPI CARDS
# =============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Est. Grid Import", "{:.0f} units/mo".format(snap.grid_import),
              delta="Protected" if snap.is_protected else "Unprotected",
              delta_color="normal" if snap.is_protected else "inverse")
with c2:
    st.metric("Metered Avg (14-mo)", "{:.0f} units".format(hist["avg_lesco"]))
with c3:
    st.metric("Solar Generation (est.)", "{:.0f} kWh/mo".format(snap.solar_monthly_kwh),
              delta="{:.0f} direct + {:.0f} battery".format(
                  snap.solar_direct_monthly, snap.solar_to_battery_monthly))
with c4:
    st.metric("Curtailed Solar", "{:.0f} kWh".format(snap.solar_curtailed_monthly),
              delta="wasted (no export)", delta_color="off")

# =============================================================
# GAUGE
# =============================================================
st.markdown("### 🎯 Protected Threshold (<={:.0f} units)".format(tariff.protected_limit))
col_g, col_i = st.columns([2, 1])

with col_g:
    gauge_max = max(300, tariff.protected_limit * 1.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=snap.grid_import,
        number={"suffix": " units", "font": {"size": 40, "color": COLORS["primary"]}},
        delta={"reference": tariff.protected_limit,
               "increasing": {"color": COLORS["danger"]},
               "decreasing": {"color": COLORS["success"]}},
        gauge={
            "axis": {"range": [0, gauge_max], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, tariff.protected_limit * 0.75], "color": "#0D2818"},
                {"range": [tariff.protected_limit * 0.75, tariff.protected_limit], "color": "#2D2800"},
                {"range": [tariff.protected_limit, gauge_max], "color": "#2D0000"},
            ],
            "threshold": {"line": {"color": COLORS["danger"], "width": 4},
                          "thickness": 0.75, "value": tariff.protected_limit},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]})
    st.plotly_chart(fig, use_container_width=True)

with col_i:
    if snap.remaining > 20:
        st.success("ON TRACK — {:.0f} units remaining".format(snap.remaining))
    elif snap.remaining > 0:
        st.warning("CAUTION — Only {:.0f} units left".format(snap.remaining))
    else:
        st.error("EXCEEDED by {:.0f} units".format(abs(snap.remaining)))

# =============================================================
# SANKEY
# =============================================================
st.markdown("### 🔄 Energy Flow (Captive — No Export)")
fig_s = go.Figure(go.Sankey(
    node=dict(pad=20, thickness=25,
              line=dict(color=COLORS["border"], width=0.5),
              label=["Grid Import", "Solar PV", "Battery", "Home Load", "Curtailed"],
              color=[COLORS["grid"], COLORS["solar"], COLORS["battery"],
                     COLORS["danger"], COLORS["muted"]]),
    link=dict(
        source=[0, 1, 1, 2, 1], target=[3, 3, 2, 3, 4],
        value=[snap.grid_import, snap.solar_direct_monthly,
               snap.solar_to_battery_monthly, snap.battery_discharge_monthly,
               snap.solar_curtailed_monthly],
        color=["rgba(88,166,255,0.3)", "rgba(255,223,74,0.3)",
               "rgba(57,211,83,0.3)", "rgba(57,211,83,0.3)",
               "rgba(139,148,158,0.3)"],
    ),
))
fig_s.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": COLORS["text"], "size": 12},
                    margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_s, use_container_width=True)

# =============================================================
# DONUT + STACKED
# =============================================================
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("### 🔌 Appliance Breakdown (Estimate)")
    fig_d = go.Figure(go.Pie(
        labels=[a.name for a in snap.appliances],
        values=[a.monthly_kwh for a in snap.appliances],
        hole=0.55, marker=dict(colors=px.colors.qualitative.Bold),
        textinfo="label+percent", textfont=dict(color=COLORS["text"], size=10),
    ))
    fig_d.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": COLORS["text"]}, showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_d, use_container_width=True)

with c_right:
    st.markdown("### 📈 14-Month Energy Breakdown")
    if hist_lesco and hist_solar_used:
        n = min(len(hist_lesco), len(hist_solar_used))
        months = df["Month"].tolist()[:n]
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=months, y=hist_solar_used[:n],
                                name="Solar Used", marker_color=COLORS["solar"]))
        fig_t.add_trace(go.Bar(x=months, y=hist_lesco[:n],
                                name="LESCO Import", marker_color=COLORS["grid"]))
        fig_t.add_hline(y=200, line_dash="dash", line_color=COLORS["danger"],
                        annotation_text="200-unit limit")
        fig_t.update_layout(barmode="stack", height=420,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": COLORS["text"]},
                            xaxis=dict(gridcolor="#21262D"),
                            yaxis=dict(gridcolor="#21262D", title="kWh"),
                            legend=dict(orientation="h", y=1.1, x=0),
                            margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Fill the history table in the sidebar.")

if hist_solar_gen and hist_solar_used:
    st.markdown("### ☀️ Solar Generation vs. Utilization")
    n = min(len(hist_solar_gen), len(hist_solar_used))
    months = df["Month"].tolist()[:n]
    fig_u = go.Figure()
    fig_u.add_trace(go.Bar(x=months, y=hist_solar_gen[:n],
                            name="Generated", marker_color=COLORS["solar"], opacity=0.5))
    fig_u.add_trace(go.Bar(x=months, y=hist_solar_used[:n],
                            name="Used (on-site)", marker_color=COLORS["success"]))
    fig_u.update_layout(barmode="overlay", height=320,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": COLORS["text"]},
                        xaxis=dict(gridcolor="#21262D"),
                        yaxis=dict(gridcolor="#21262D", title="kWh"),
                        legend=dict(orientation="h", y=1.1, x=0),
                        margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_u, use_container_width=True)

# =============================================================
# RECOMMENDATIONS
# =============================================================
st.markdown("---")
st.markdown("### 🎯 Top Recommendations")

recs = generate_recommendations(snap, hist)
level_class = {"danger": "rec-danger", "warning": "rec-warning",
               "info": "rec-info", "success": "rec-success"}
level_icon = {"danger": "🚨", "warning": "⚠️", "info": "💡", "success": "✅"}

focus_order = ["INCREASE SOLAR?", "INCREASE BATTERY?",
               "2nd LESCO METER?", "LOAD OPTIMIZATION"]
for focus in focus_order:
    for r in [x for x in recs if x["focus"] == focus]:
        st.markdown(
            '<div class="rec-card {}">'
            '<div class="rec-focus">◆ {}</div>'
            '<div class="rec-title">{} {}</div>'
            '<div class="rec-msg">{}</div>'
            '<div class="rec-action">→ {}</div>'
            '</div>'.format(
                level_class.get(r["level"], "rec-info"),
                r["focus"], level_icon.get(r["level"], ""),
                r["title"], r["message"], r["action"],
            ),
            unsafe_allow_html=True,
        )

# =============================================================
# LEGAL WARNING
# =============================================================
st.markdown("---")
st.warning(
    "**Legal Notice (September 2026):** LESCO is actively cracking down on "
    "multiple meters used to artificially stay under 200 units. Households "
    "found doing this immediately lose protected status. Multiple meters are "
    "only permissible where **separate families occupy distinct portions with "
    "independent kitchens and wiring**."
)

# =============================================================
# BILL BREAKDOWN
# =============================================================
st.markdown("### 💰 Bill Breakdown (Estimated)")
b = snap.bill
bc = st.columns(4)
with bc[0]:
    st.metric("Protected ({} units)".format(b["p_units"]),
              "PKR {:,.0f}".format(b["p_cost"]),
              delta="@ {}/unit".format(tariff.protected_rate), delta_color="off")
with bc[1]:
    if b["u_units"] > 0:
        st.metric("Unprotected ({} units)".format(b["u_units"]),
                  "PKR {:,.0f}".format(b["u_cost"]),
                  delta="@ {}/unit".format(tariff.unprotected_rate), delta_color="off")
    else:
        st.metric("Unprotected", "PKR 0", delta="none", delta_color="off")
with bc[2]:
    st.metric("Effective Rate", "PKR {:.2f}/unit".format(b["per_unit"]))
with bc[3]:
    st.metric("Total Bill", "PKR {:,.0f}".format(b["total"]))

# =============================================================
# AI — SOLAR USAGE STRATEGY
# =============================================================
st.markdown("---")
st.markdown("### 🤖 AI Solar Usage Strategy (Groq / openai/gpt-oss-120b)")
st.caption(
    "The AI receives pre-computed generation/load numbers, so it cannot "
    "hallucinate kWh values — it only recommends strategy."
)

if st.button("Run AI Strategy", use_container_width=True):
    with st.spinner("Asking Groq..."):
        st.session_state.ai_result = run_solar_strategy(snap, hist)
if st.session_state.ai_result:
    st.markdown(st.session_state.ai_result)

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8B949E; font-size:0.8rem;'>"
    "Home Energy Manager v2.2 | Captive solar (no export) | "
    "Formula-based solar estimate | AI strategy only"
    "</p>",
    unsafe_allow_html=True,
)
