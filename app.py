"""
Home Energy Management Dashboard
LESCO Protected Consumer Optimizer - September 2026

Single-file app. Captive solar (NO export to grid). Surplus is curtailed.
Appliances fully editable. 12-month history inputs.
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

LAHORE_PEAK_SUN_HOURS = 5.2
DAY_LOAD_RATIO = 0.55
BATTERY_EFFICIENCY = 0.85

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Default rolling window: Sep 2025 -> Aug 2026
DEFAULT_MONTH_SEQ = ["Sep-25", "Oct-25", "Nov-25", "Dec-25",
                     "Jan-26", "Feb-26", "Mar-26", "Apr-26",
                     "May-26", "Jun-26", "Jul-26", "Aug-26"]

COLORS = {
    "bg": "#0D1117", "bg2": "#161B22", "border": "#30363D",
    "text": "#C9D1D9", "muted": "#8B949E",
    "primary": "#58A6FF", "success": "#3FB950",
    "warning": "#D29922", "danger": "#F85149",
    "solar": "#FFDF4A", "battery": "#39D353",
    "grid": "#58A6FF", "waste": "#8B949E",
}

# =============================================================
# DEFAULT APPLIANCES  (watts, hours per session, days/week, qty)
# -------------------------------------------------------------
# watts = running average (inverter ACs modulate; not peak nameplate)
# hours = hours per ACTIVE day
# days_per_week = 1..7
# =============================================================
DEFAULT_APPLIANCES = [
    {"name": "Inverter AC 1.5T @26C (2x/week)", "watts": 550,  "hours": 6.0,  "days_per_week": 2, "qty": 1},
    {"name": "Inverter AC 1.5T @26C (daily)",   "watts": 550,  "hours": 16.0, "days_per_week": 7, "qty": 1},
    {"name": "Inverter AC 1.0T @26C (daily)",   "watts": 420,  "hours": 16.0, "days_per_week": 7, "qty": 1},
    {"name": "Microwave Oven",                  "watts": 1200, "hours": 0.25, "days_per_week": 7, "qty": 1},
    {"name": "Washing Machine (weekly)",        "watts": 500,  "hours": 4.0,  "days_per_week": 1, "qty": 1},
    {"name": "Submersible Pump",                "watts": 750,  "hours": 2.0,  "days_per_week": 7, "qty": 1},
    {"name": "Machine 3kW (2x/week)",           "watts": 3000, "hours": 5.0,  "days_per_week": 2, "qty": 1},
    {"name": "RO Plant 1.5kW (2x/week)",        "watts": 1500, "hours": 5.0,  "days_per_week": 2, "qty": 1},
    {"name": "BLDC Ceiling Fan",                "watts": 28,   "hours": 12.0, "days_per_week": 7, "qty": 6},
    {"name": "LED Light 12W",                   "watts": 12,   "hours": 6.0,  "days_per_week": 7, "qty": 30},
    {"name": "Philips Iron (weekly)",           "watts": 1000, "hours": 3.0,  "days_per_week": 1, "qty": 1},
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
            "p_units": round(p_units),
            "u_units": round(u_units),
            "p_cost": round(p_cost),
            "u_cost": round(u_cost),
        }


@dataclass
class Snapshot:
    appliances: list
    system: SystemConfig
    tariff: Tariff
    # Load
    total_daily_kwh: float = 0.0
    total_monthly_kwh: float = 0.0
    day_load_daily: float = 0.0
    night_load_daily: float = 0.0
    # Solar
    solar_daily_kwh: float = 0.0
    solar_monthly_kwh: float = 0.0
    solar_direct_monthly: float = 0.0
    solar_to_battery_monthly: float = 0.0
    solar_curtailed_monthly: float = 0.0   # wasted (no export)
    # Battery
    battery_discharge_monthly: float = 0.0
    battery_usable_daily: float = 0.0
    # Grid
    grid_import: float = 0.0
    # Status
    is_protected: bool = False
    remaining: float = 0.0
    bill: dict = field(default_factory=dict)


# =============================================================
# AUTO-CALIBRATION
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
    has_kwh = len(kwh) >= 3
    has_bills = len(bills) >= 3

    if has_kwh and has_bills and len(kwh) == len(bills):
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
    elif has_kwh and not has_bills:
        t.source = "defaults (kWh history: {} months)".format(len(kwh))
    elif has_bills and not has_kwh:
        avg_bill = sum(bills) / len(bills)
        t.source = "defaults (avg bill: PKR {:,.0f})".format(avg_bill)
    return t


# =============================================================
# ENERGY COMPUTATION (captive solar - no export)
# -------------------------------------------------------------
# Daily flow:
#   1. Solar serves day-load first (solar_direct)
#   2. Surplus charges battery (capped by battery capacity)
#   3. Any remaining surplus is CURTAILED (wasted)
#   4. Battery discharges at night (after round-trip losses)
#   5. Grid covers whatever solar-direct + battery cannot
# =============================================================
def build_snapshot(appliances, system, tariff):
    s = Snapshot(appliances=appliances, system=system, tariff=tariff)

    s.total_daily_kwh = sum(a.daily_avg_kwh for a in appliances)
    s.total_monthly_kwh = s.total_daily_kwh * 30

    s.day_load_daily = s.total_daily_kwh * DAY_LOAD_RATIO
    s.night_load_daily = s.total_daily_kwh * (1.0 - DAY_LOAD_RATIO)

    s.solar_daily_kwh = system.solar_kwp * LAHORE_PEAK_SUN_HOURS
    s.solar_monthly_kwh = s.solar_daily_kwh * 30

    # 1. Direct solar to day-load
    solar_direct = min(s.solar_daily_kwh, s.day_load_daily)

    # 2. Surplus solar -> battery (capped)
    solar_surplus = max(0.0, s.solar_daily_kwh - solar_direct)
    battery_charge_in = min(solar_surplus, system.battery_kwh)

    # 3. Remaining surplus is CURTAILED (no grid export)
    solar_curtailed = solar_surplus - battery_charge_in

    # 4. Battery discharge (with round-trip losses)
    battery_usable = battery_charge_in * BATTERY_EFFICIENCY
    battery_discharge = min(battery_usable, s.night_load_daily)

    # 5. Grid import
    grid_daily = max(0.0, s.total_daily_kwh - solar_direct - battery_discharge)

    s.solar_direct_monthly = solar_direct * 30
    s.solar_to_battery_monthly = battery_charge_in * 30
    s.solar_curtailed_monthly = solar_curtailed * 30
    s.battery_discharge_monthly = battery_discharge * 30
    s.battery_usable_daily = battery_usable
    s.grid_import = grid_daily * 30

    s.is_protected = s.grid_import <= tariff.protected_limit
    s.remaining = tariff.protected_limit - s.grid_import
    s.bill = tariff.bill_for(s.grid_import)
    return s


# =============================================================
# RECOMMENDATIONS ENGINE
# =============================================================
def generate_recommendations(snap, hist):
    """
    Returns list of dicts: {level, title, message, action}
    levels: 'danger' | 'warning' | 'info' | 'success'
    """
    recs = []

    # --- History-derived metrics ---
    avg_consumption = hist["avg_consumption"]
    avg_solar_gen = hist["avg_solar_gen"]
    avg_solar_used = hist["avg_solar_used"]
    avg_lesco = hist["avg_lesco"]

    # 1. LESCO import threshold
    if avg_lesco > 200:
        shortfall = avg_lesco - 200
        extra_kwp = shortfall / (LAHORE_PEAK_SUN_HOURS * 30)
        recs.append({
            "level": "danger",
            "title": "Exceeding protected limit by {:.0f} units/month".format(shortfall),
            "message": (
                "Your average LESCO import is {:.0f} units. To stay under 200, "
                "you need approximately {:.1f} kWp of additional solar "
                "(assuming it's fully utilized).".format(avg_lesco, extra_kwp)
            ),
            "action": "Increase solar PV by ~{:.1f} kWp".format(extra_kwp),
        })
    elif avg_lesco > 180:
        recs.append({
            "level": "warning",
            "title": "Close to 200-unit threshold",
            "message": "Average LESCO import is {:.0f} units. Small increases could push you over.".format(avg_lesco),
            "action": "Add 1-2 kWp solar or reduce load during peak months",
        })
    else:
        recs.append({
            "level": "success",
            "title": "Comfortably under 200 units",
            "message": "Average LESCO import is {:.0f} units. You have headroom.".format(avg_lesco),
            "action": "Maintain current usage or optimize for savings",
        })

    # 2. Solar utilization
    if avg_solar_gen > 0 and avg_solar_used > 0:
        util_pct = (avg_solar_used / avg_solar_gen) * 100.0
        if util_pct < 70:
            wasted = avg_solar_gen - avg_solar_used
            ideal_battery = snap.night_load_daily / BATTERY_EFFICIENCY
            recs.append({
                "level": "warning",
                "title": "Solar utilization is only {:.0f}%".format(util_pct),
                "message": (
                    "You generate {:.0f} units but only use {:.0f}. About {:.0f} units/month "
                    "are being wasted because there is no export. "
                    "Increase battery to capture this surplus.".format(
                        avg_solar_gen, avg_solar_used, wasted)
                ),
                "action": "Increase battery to ≥ {:.0f} kWh".format(ideal_battery),
            })
        else:
            recs.append({
                "level": "success",
                "title": "Solar utilization is {:.0f}%".format(util_pct),
                "message": "Most of your solar generation is being consumed on-site.",
                "action": "No change needed for solar utilization",
            })

    # 3. Battery sizing
    ideal_battery = snap.night_load_daily / BATTERY_EFFICIENCY
    if snap.system.battery_kwh < ideal_battery * 0.8:
        recs.append({
            "level": "warning",
            "title": "Battery undersized",
            "message": (
                "Your night load is ~{:.1f} kWh/night. Ideal battery is ~{:.1f} kWh "
                "but you have {:.1f} kWh. Increase battery to shift more night load "
                "off grid.".format(snap.night_load_daily, ideal_battery, snap.system.battery_kwh)
            ),
            "action": "Increase battery to ~{:.0f} kWh".format(ideal_battery),
        })
    elif snap.system.battery_kwh > ideal_battery * 1.4:
        recs.append({
            "level": "info",
            "title": "Battery may be oversized",
            "message": (
                "Your battery ({:.0f} kWh) exceeds your night load ({:.1f} kWh). "
                "Some battery capacity may be unused.".format(
                    snap.system.battery_kwh, snap.night_load_daily)
            ),
            "action": "Consider if a smaller battery would suffice",
        })

    # 4. 2nd LESCO meter
    if avg_lesco > 200 and avg_lesco < 450:
        recs.append({
            "level": "warning",
            "title": "2nd LESCO meter - legal risk",
            "message": (
                "A 2nd meter is only legal for separate families with independent "
                "kitchens and wiring. LESCO actively cracks down on splitting loads. "
                "Solar + battery is the safer path."
            ),
            "action": "Prefer solar/battery expansion over 2nd meter",
        })
    elif avg_lesco >= 450:
        recs.append({
            "level": "info",
            "title": "Consider load separation",
            "message": (
                "Your consumption is very high. If you have a genuinely separate "
                "portion of the home (kitchen + wiring), a 2nd meter may be "
                "justifiable. Otherwise add solar capacity instead."
            ),
            "action": "Evaluate legal 2nd meter vs. more solar",
        })

    # 5. Load optimization - biggest consumer
    if snap.appliances:
        biggest = max(snap.appliances, key=lambda a: a.monthly_kwh)
        recs.append({
            "level": "info",
            "title": "Load optimization: {}".format(biggest.name),
            "message": (
                "Your largest load is '{}' at {:.0f} kWh/month ({:.0f}% of total). "
                "Shift this to 10 AM-3 PM to use solar directly.".format(
                    biggest.name, biggest.monthly_kwh,
                    (biggest.monthly_kwh / max(snap.total_monthly_kwh, 1)) * 100)
            ),
            "action": "Shift {} to solar hours (10 AM - 3 PM)".format(biggest.name),
        })

    # 6. Appliance estimate vs metered
    if avg_consumption > 0 and snap.total_monthly_kwh > 0:
        diff_pct = ((snap.total_monthly_kwh - avg_consumption) / avg_consumption) * 100.0
        if abs(diff_pct) > 20:
            recs.append({
                "level": "warning",
                "title": "Appliance estimate off by {:.0f}%".format(abs(diff_pct)),
                "message": (
                    "Appliance-based estimate is {:.0f} kWh/month but your metered "
                    "average is {:.0f} kWh/month. Adjust watts/hours in the sidebar "
                    "to match reality.".format(snap.total_monthly_kwh, avg_consumption)
                ),
                "action": "Fine-tune appliance wattages",
            })

    return recs


# =============================================================
# AI ADVISOR
# =============================================================
def run_ai_advisor(appliance_summary: str, solar_kwp: float,
                   battery_kwh: float, monthly_units: float,
                   avg_consumption: float, avg_solar_gen: float,
                   avg_solar_used: float) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return "AI advisor is disabled. Add GROQ_API_KEY to Streamlit Secrets."

    system_prompt = (
        "You are an expert energy advisor for a household in Lahore, Pakistan. "
        "The household uses a CAPTIVE solar + battery system with NO grid export "
        "(no net metering). Any surplus solar is curtailed. The household is on "
        "the LESCO protected-consumer tariff (under 200 units/month). "
        "Give concise, actionable advice. No markdown headers."
    )
    user_prompt = (
        "Appliance load: {}.\n"
        "Solar: {} kWp. Battery: {} kWh (solar-charged only).\n"
        "Current grid import: {:.0f} units/month.\n"
        "Historical avg consumption: {:.0f} units/month.\n"
        "Historical avg solar generation: {:.0f} units/month.\n"
        "Historical avg solar actually used: {:.0f} units/month.\n"
        "1. Assess solar utilization.\n"
        "2. Recommend: increase solar? increase battery? 2nd meter?\n"
        "3. Give 3 load-optimization tips."
    ).format(appliance_summary, solar_kwp, battery_kwh, monthly_units,
             avg_consumption, avg_solar_gen, avg_solar_used)

    try:
        r = requests.post(
            GROQ_API_URL,
            headers={"Authorization": "Bearer " + api_key,
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}],
                  "temperature": 0.7, "max_tokens": 1200},
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
        return "AI analysis failed: HTTP {}".format(code)
    except Exception as e:
        return "AI analysis failed: {}".format(e)


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
    .action-card {
        background: #161B22; border-left: 5px solid #3FB950;
        border-radius: 12px; padding: 1rem 1.5rem;
        margin: 0.5rem 0; color: #C9D1D9;
    }
    .action-card-warning { border-left-color: #D29922; }
    .action-card-danger  { border-left-color: #F85149; }
    .action-card-info    { border-left-color: #58A6FF; }
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
        margin: 0.5rem 0; color: #C9D1D9;
    }
    .rec-danger  { border-left-color: #F85149; }
    .rec-warning { border-left-color: #D29922; }
    .rec-success { border-left-color: #3FB950; }
    .rec-info    { border-left-color: #58A6FF; }
    .rec-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 0.25rem; }
    .rec-msg   { font-size: 0.9rem; color: #8B949E; margin-bottom: 0.4rem; }
    .rec-action{ font-size: 0.85rem; color: #3FB950; font-weight: 600; }
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
        "Consumption (kWh)": [None] * 12,
        "Solar Gen (kWh)":   [None] * 12,
        "Solar Used (kWh)":  [None] * 12,
        "LESCO Import (kWh)": [None] * 12,
        "Bill (PKR)":        [None] * 12,
    })
if "ai_result" not in st.session_state:
    st.session_state.ai_result = ""

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## 🔌 Appliance Manager")
    st.caption("All fields editable. `Days/week` = 7 for daily appliances.")
    st.markdown("---")

    with st.expander("➕ Add New Appliance", expanded=False):
        n_name = st.text_input("Appliance Name", key="nn")
        c1, c2, c3, c4 = st.columns(4)
        n_w = c1.number_input("Watts", min_value=1, max_value=20000,
                              value=100, step=10, key="nw")
        n_h = c2.number_input("Hrs/session", min_value=0.1, max_value=24.0,
                              value=1.0, step=0.25, key="nh")
        n_d = c3.number_input("Days/week", min_value=1, max_value=7,
                              value=7, step=1, key="nd")
        n_q = c4.number_input("Qty", min_value=1, max_value=50,
                              value=1, step=1, key="nq")
        if st.button("Add Appliance", use_container_width=True) and n_name:
            st.session_state.appliances.append({
                "name": n_name, "watts": int(n_w), "hours": float(n_h),
                "days_per_week": int(n_d), "qty": int(n_q),
            })
            st.rerun()

    st.markdown("### Current Appliances")
    for i, app in enumerate(st.session_state.appliances):
        label = "{} — {}W × {} d/wk".format(app["name"], app["watts"], app["days_per_week"])
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
        '<div class="captive-note">🔒 <b>Captive system</b> — no export to grid. '
        'Surplus solar is curtailed.</div>',
        unsafe_allow_html=True,
    )
    st.session_state.solar_kwp = st.slider(
        "Solar PV System (kWp)", 0.0, 30.0, st.session_state.solar_kwp, 0.5)
    st.session_state.battery_kwh = st.slider(
        "Battery Capacity (kWh)", 0.0, 80.0, st.session_state.battery_kwh, 1.0)

    st.markdown("---")
    st.markdown("## 📊 12-Month History")
    st.caption("Edit cells directly. Leave blank if unknown.")
    st.session_state.hist_df = st.data_editor(
        st.session_state.hist_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="hist_editor",
    )

# =============================================================
# BUILD SNAPSHOT
# =============================================================
appliances = [Appliance(**a) for a in st.session_state.appliances]
system = SystemConfig(solar_kwp=st.session_state.solar_kwp,
                      battery_kwh=st.session_state.battery_kwh)

# Derive tariff from history (LESCO import vs Bill columns)
df = st.session_state.hist_df.copy()
def _clean(series):
    return pd.to_numeric(series, errors="coerce").dropna().tolist()

hist_lesco = _clean(df["LESCO Import (kWh)"])
hist_bills = _clean(df["Bill (PKR)"])
hist_consumption = _clean(df["Consumption (kWh)"])
hist_solar_gen = _clean(df["Solar Gen (kWh)"])
hist_solar_used = _clean(df["Solar Used (kWh)"])

tariff = calibrate_tariff(
    ",".join(map(str, hist_lesco)),
    ",".join(map(str, hist_bills)),
)

snap = build_snapshot(appliances, system, tariff)

# History metrics
hist = {
    "avg_consumption": sum(hist_consumption) / len(hist_consumption) if hist_consumption else snap.total_monthly_kwh,
    "avg_solar_gen": sum(hist_solar_gen) / len(hist_solar_gen) if hist_solar_gen else snap.solar_monthly_kwh,
    "avg_solar_used": sum(hist_solar_used) / len(hist_solar_used) if hist_solar_used else snap.solar_direct_monthly,
    "avg_lesco": sum(hist_lesco) / len(hist_lesco) if hist_lesco else snap.grid_import,
    "avg_bill": sum(hist_bills) / len(hist_bills) if hist_bills else snap.bill["total"],
}

# =============================================================
# HEADER
# =============================================================
st.markdown('<div class="main-header">Home Energy Management Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#8B949E;'>"
    "September 2026 | LESCO Protected Consumer Optimizer | "
    "Captive Solar (No Export)</p>",
    unsafe_allow_html=True,
)

badge_class = "tariff-badge tariff-badge-ok" if tariff.calibrated else "tariff-badge"
badge_prefix = "OK " if tariff.calibrated else "DEFAULTS "
badge_text = (
    badge_prefix + "Tariff: " + tariff.source
    + " | Protected <=" + str(int(tariff.protected_limit))
    + " @ PKR " + str(tariff.protected_rate) + "/unit"
    + " | Unprotected @ PKR " + str(tariff.unprotected_rate) + "/unit"
)
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
    st.metric("Monthly Grid Units (est.)", "{:.0f}".format(snap.grid_import),
              delta="Protected" if snap.is_protected else "Unprotected",
              delta_color="normal" if snap.is_protected else "inverse")
with c2:
    st.metric("Metered Avg (12-mo)", "{:.0f} units".format(hist["avg_lesco"]),
              delta="from history")
with c3:
    st.metric("Solar Generation", "{:.0f} kWh".format(snap.solar_monthly_kwh),
              delta="{:.0f} direct + {:.0f} battery".format(
                  snap.solar_direct_monthly, snap.solar_to_battery_monthly))
with c4:
    st.metric("Solar Curtailed", "{:.0f} kWh".format(snap.solar_curtailed_monthly),
              delta="wasted (no export)",
              delta_color="off")

# =============================================================
# GAUGE + STATUS
# =============================================================
st.markdown("### 🎯 Protected Status Threshold (<={:.0f} units)".format(tariff.protected_limit))
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
        st.success("ON TRACK\n\n{:.0f} units remaining".format(snap.remaining))
    elif snap.remaining > 0:
        st.warning("CAUTION\n\nOnly {:.0f} units left".format(snap.remaining))
    else:
        st.error("EXCEEDED by {:.0f} units".format(abs(snap.remaining)))

# =============================================================
# SANKEY - no export
# =============================================================
st.markdown("### 🔄 Energy Flow (Captive — No Export)")
fig_s = go.Figure(go.Sankey(
    node=dict(pad=20, thickness=25,
              line=dict(color=COLORS["border"], width=0.5),
              label=["Grid Import", "Solar PV", "Battery", "Home Load", "Curtailed"],
              color=[COLORS["grid"], COLORS["solar"], COLORS["battery"],
                     COLORS["danger"], COLORS["muted"]]),
    link=dict(
        source=[0, 1, 1, 2, 1],
        target=[3, 3, 2, 3, 4],
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
# DONUT + 12-MONTH STACKED
# =============================================================
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("### 🔌 Appliance Breakdown (Estimate)")
    fig_d = go.Figure(go.Pie(
        labels=[a.name for a in snap.appliances],
        values=[a.monthly_kwh for a in snap.appliances],
        hole=0.55, marker=dict(colors=px.colors.qualitative.Bold),
        textinfo="label+percent", textfont=dict(color=COLORS["text"], size=11),
    ))
    fig_d.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": COLORS["text"]}, showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_d, use_container_width=True)

with c_right:
    st.markdown("### 📈 12-Month Energy Breakdown")
    if hist_lesco and hist_solar_used:
        months = df["Month"].tolist()[:len(hist_lesco)]
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(
            x=months, y=hist_solar_used, name="Solar Used",
            marker_color=COLORS["solar"],
        ))
        fig_t.add_trace(go.Bar(
            x=months, y=hist_lesco, name="LESCO Import",
            marker_color=COLORS["grid"],
        ))
        fig_t.add_hline(y=200, line_dash="dash", line_color=COLORS["danger"],
                        annotation_text="Protected Limit (200)")
        fig_t.update_layout(
            barmode="stack", height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text"]},
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D", title="kWh"),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Fill in the 12-month history table in the sidebar to see the breakdown.")

# Solar utilization chart
if hist_solar_gen and hist_solar_used:
    st.markdown("### ☀️ Solar Generation vs. Utilization")
    months = df["Month"].tolist()[:len(hist_solar_gen)]
    used = hist_solar_used[:len(hist_solar_gen)]
    gen = hist_solar_gen[:len(hist_solar_gen)]
    fig_u = go.Figure()
    fig_u.add_trace(go.Bar(x=months, y=gen, name="Generated",
                            marker_color=COLORS["solar"], opacity=0.5))
    fig_u.add_trace(go.Bar(x=months, y=used, name="Used (on-site)",
                            marker_color=COLORS["success"]))
    fig_u.update_layout(
        barmode="overlay", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        xaxis=dict(gridcolor="#21262D"),
        yaxis=dict(gridcolor="#21262D", title="kWh"),
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_u, use_container_width=True)

# =============================================================
# TOP RECOMMENDATIONS
# =============================================================
st.markdown("---")
st.markdown("### 🎯 Top Recommendations")

recs = generate_recommendations(snap, hist)
level_class = {"danger": "rec-danger", "warning": "rec-warning",
               "info": "rec-info", "success": "rec-success"}
level_icon = {"danger": "🚨", "warning": "⚠️", "info": "💡", "success": "✅"}

for r in recs:
    st.markdown(
        '<div class="rec-card {}">'
        '<div class="rec-title">{} {}</div>'
        '<div class="rec-msg">{}</div>'
        '<div class="rec-action">→ {}</div>'
        '</div>'.format(
            level_class.get(r["level"], "rec-info"),
            level_icon.get(r["level"], ""),
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
# AI INSIGHTS
# =============================================================
st.markdown("---")
st.markdown("### 🤖 AI Energy Advisor (Groq / openai/gpt-oss-120b)")

if st.button("Run AI Analysis", use_container_width=True):
    summary = ", ".join(
        "{} ({}W × {}h × {}d/wk × {})".format(a.name, a.watts, a.hours,
                                              a.days_per_week, a.qty)
        for a in snap.appliances
    )
    with st.spinner("Asking Groq..."):
        st.session_state.ai_result = run_ai_advisor(
            appliance_summary=summary,
            solar_kwp=snap.system.solar_kwp,
            battery_kwh=snap.system.battery_kwh,
            monthly_units=snap.grid_import,
            avg_consumption=hist["avg_consumption"],
            avg_solar_gen=hist["avg_solar_gen"],
            avg_solar_used=hist["avg_solar_used"],
        )
if st.session_state.ai_result:
    st.markdown(st.session_state.ai_result)

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8B949E; font-size:0.8rem;'>"
    "Home Energy Manager v2.0 | Captive solar (no export) | "
    "Editable appliances | 12-month history | Direct Groq API"
    "</p>",
    unsafe_allow_html=True,
)
