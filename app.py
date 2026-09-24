"""
Home Energy Management Dashboard
LESCO Protected Consumer Optimizer - September 2026

The app auto-calibrates the tariff structure from the user's
previous-year monthly kWh and/or monthly PKR bills.
No manual entry of tariffs, rates, or thresholds required.
"""
import os

# --- Bridge Streamlit secrets to os.environ (required for CrewAI/litellm) ---
try:
    for _k in ("GROQ_API_KEY",):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k]).strip()
except Exception:
    pass
# ---------------------------------------------------------------------------
"""
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field

# =============================================================
# 1. INTERNAL DEFAULTS (used only as fallback when no history)
# =============================================================
DEFAULT_PROTECTED_LIMIT = 200.0
DEFAULT_PROTECTED_RATE = 13.40
DEFAULT_UNPROTECTED_RATE = 42.80
DEFAULT_SOLAR_KWP = 5.0
DEFAULT_BATTERY_KWH = 10.0

LAHORE_PEAK_SUN_HOURS = 5.2
DAY_LOAD_RATIO = 0.60
BATTERY_EFFICIENCY = 0.85
BATTERY_SHIFT_LIMIT = 0.40
EXPORT_RATE = 11.0

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COLORS = {
    "bg": "#0D1117", "bg2": "#161B22", "border": "#30363D",
    "text": "#C9D1D9", "muted": "#8B949E",
    "primary": "#58A6FF", "success": "#3FB950",
    "warning": "#D29922", "danger": "#F85149",
    "solar": "#FFDF4A", "battery": "#39D353",
}

DEFAULT_APPLIANCES = [
    {"name": "Air Conditioner (1.5 Ton)", "watts": 1500, "hours": 8.0, "qty": 1},
    {"name": "Refrigerator", "watts": 200, "hours": 24.0, "qty": 1},
    {"name": "LED Lights (set of 10)", "watts": 100, "hours": 6.0, "qty": 1},
    {"name": "Ceiling Fan", "watts": 80, "hours": 12.0, "qty": 3},
    {"name": "Water Heater (Geyser)", "watts": 2000, "hours": 1.0, "qty": 1},
    {"name": "Washing Machine", "watts": 500, "hours": 1.0, "qty": 1},
]

# =============================================================
# 2. DATA MODELS
# =============================================================
@dataclass
class Appliance:
    name: str
    watts: float
    hours: float
    qty: int = 1

    @property
    def daily_kwh(self) -> float:
        return (self.watts * self.hours * self.qty) / 1000

    @property
    def monthly_kwh(self) -> float:
        return self.daily_kwh * 30


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
    confidence: str = "low"

    def bill_for(self, units: float) -> dict:
        if units <= 0:
            return {"total": 0, "per_unit": 0,
                    "p_units": 0, "u_units": 0, "p_cost": 0, "u_cost": 0}
        p_units = min(units, self.protected_limit)
        u_units = max(0.0, units - self.protected_limit)
        p_cost = p_units * self.protected_rate
        u_cost = u_units * self.unprotected_rate
        total = p_cost + u_cost
        return {
            "total": round(total),
            "per_unit": round(total / units, 2),
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
    total_daily_kwh: float = 0.0
    total_monthly_kwh: float = 0.0
    solar_monthly_kwh: float = 0.0
    solar_self_consumed: float = 0.0
    solar_export: float = 0.0
    export_credit: float = 0.0
    battery_shifted: float = 0.0
    grid_before_battery: float = 0.0
    grid_after_battery: float = 0.0
    is_protected: bool = False
    remaining: float = 0.0
    bill: dict = field(default_factory=dict)

# =============================================================
# 3. AUTO-CALIBRATION
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
        t.confidence = "high" if len(kwh) >= 8 else "medium"

    elif has_kwh and not has_bills:
        t.source = "defaults (kWh history: {} months)".format(len(kwh))
        t.confidence = "low"

    elif has_bills and not has_kwh:
        avg_bill = sum(bills) / len(bills)
        t.source = "defaults (avg bill: PKR {:,.0f})".format(avg_bill)
        t.confidence = "low"

    return t

# =============================================================
# 4. ENERGY COMPUTATION
# =============================================================
def build_snapshot(appliances, system, tariff):
    s = Snapshot(appliances=appliances, system=system, tariff=tariff)

    s.total_daily_kwh = sum(a.daily_kwh for a in appliances)
    s.total_monthly_kwh = s.total_daily_kwh * 30

    s.solar_monthly_kwh = system.solar_kwp * LAHORE_PEAK_SUN_HOURS * 30
    day_load = s.total_monthly_kwh * DAY_LOAD_RATIO
    s.solar_self_consumed = min(s.solar_monthly_kwh, day_load)
    s.solar_export = max(0.0, s.solar_monthly_kwh - s.solar_self_consumed)
    s.export_credit = s.solar_export * EXPORT_RATE

    s.grid_before_battery = max(0.0, s.total_monthly_kwh - s.solar_self_consumed)

    raw_shift = system.battery_kwh * 30 * BATTERY_EFFICIENCY
    s.battery_shifted = min(raw_shift, s.grid_before_battery * BATTERY_SHIFT_LIMIT)

    s.grid_after_battery = max(0.0, s.grid_before_battery - s.battery_shifted)

    s.is_protected = s.grid_after_battery <= tariff.protected_limit
    s.remaining = tariff.protected_limit - s.grid_after_battery

    s.bill = tariff.bill_for(s.grid_after_battery)
    return s

# =============================================================
# 5. PAGE CONFIG + CSS
# =============================================================
st.set_page_config(
    page_title="Home Energy Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
    }
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
</style>
""", unsafe_allow_html=True)

# =============================================================
# 6. SESSION STATE
# =============================================================
if "appliances" not in st.session_state:
    st.session_state.appliances = [dict(a) for a in DEFAULT_APPLIANCES]
if "solar_kwp" not in st.session_state:
    st.session_state.solar_kwp = DEFAULT_SOLAR_KWP
if "battery_kwh" not in st.session_state:
    st.session_state.battery_kwh = DEFAULT_BATTERY_KWH
if "hist_kwh" not in st.session_state:
    st.session_state.hist_kwh = ""
if "hist_bill" not in st.session_state:
    st.session_state.hist_bill = ""
if "ai_result" not in st.session_state:
    st.session_state.ai_result = ""

# =============================================================
# 7. SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## Appliance Manager")
    st.markdown("---")

    with st.expander("Add New Appliance", expanded=False):
        new_name = st.text_input("Appliance Name", key="new_name")
        c1, c2, c3 = st.columns(3)
        new_w = c1.number_input("Watts", min_value=1, max_value=10000,
                                value=100, step=10, key="new_w")
        new_h = c2.number_input("Hours/Day", min_value=0.1, max_value=24.0,
                                value=2.0, step=0.5, key="new_h")
        new_q = c3.number_input("Qty", min_value=1, max_value=20,
                                value=1, step=1, key="new_q")
        if st.button("Add Appliance", use_container_width=True) and new_name:
            st.session_state.appliances.append({
                "name": new_name,
                "watts": int(new_w),
                "hours": float(new_h),
                "qty": int(new_q),
            })
            st.rerun()

    st.markdown("### Current Appliances")
    for i, app in enumerate(st.session_state.appliances):
        label = "{} - {}W x {}".format(app["name"], app["watts"], app["qty"])
        with st.expander(label, expanded=False):
            app["watts"] = st.number_input(
                "Watts", min_value=1, max_value=10000,
                value=int(app["watts"]), step=10, key="w_{}".format(i),
            )
            app["hours"] = st.number_input(
                "Hours/Day", min_value=0.1, max_value=24.0,
                value=float(app["hours"]), step=0.5, key="h_{}".format(i),
            )
            app["qty"] = st.number_input(
                "Quantity", min_value=1, max_value=20,
                value=int(app["qty"]), step=1, key="q_{}".format(i),
            )
            if st.button("Remove", key="del_{}".format(i)):
                st.session_state.appliances.pop(i)
                st.rerun()

    st.markdown("---")
    st.markdown("## Solar & Battery")
    st.session_state.solar_kwp = st.slider(
        "Solar PV System (kWp)", 0.0, 20.0, st.session_state.solar_kwp, 0.5,
    )
    st.session_state.battery_kwh = st.slider(
        "Battery Capacity (kWh)", 0.0, 50.0, st.session_state.battery_kwh, 1.0,
    )

    st.markdown("---")
    st.markdown("## Previous Year Data")
    st.caption(
        "Paste last 12 months of consumption and/or bills. "
        "The app auto-derives the effective tariff."
    )

    st.session_state.hist_kwh = st.text_area(
        "Monthly kWh (12 comma-separated values)",
        value=st.session_state.hist_kwh,
        placeholder="185, 192, 210, 265, 340, 410, 425, 405, 320, 245, 195, 178",
        height=80,
    )
    st.session_state.hist_bill = st.text_area(
        "Monthly PKR bills (12 comma-separated values, optional)",
        value=st.session_state.hist_bill,
        placeholder="3200, 3450, 4100, 6800, 12500, 16800, 17400, 16500, 11800, 7200, 3900, 3100",
        height=80,
    )

# =============================================================
# 8. BUILD SNAPSHOT
# =============================================================
appliances = [Appliance(**a) for a in st.session_state.appliances]
system = SystemConfig(
    solar_kwp=st.session_state.solar_kwp,
    battery_kwh=st.session_state.battery_kwh,
)
tariff = calibrate_tariff(st.session_state.hist_kwh, st.session_state.hist_bill)
snap = build_snapshot(appliances, system, tariff)

# =============================================================
# 9. HEADER
# =============================================================
st.markdown('<div class="main-header">Home Energy Management Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#8B949E;'>"
    "September 2026 | LESCO Protected Consumer Optimizer</p>",
    unsafe_allow_html=True,
)

badge_class = "tariff-badge tariff-badge-ok" if tariff.calibrated else "tariff-badge"
badge_prefix = "OK " if tariff.calibrated else "DEFAULTS "
badge_text = (
    badge_prefix
    + "Tariff: " + tariff.source
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
# 10. KPI CARDS
# =============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Monthly Grid Units", "{:.0f}".format(snap.grid_after_battery),
              delta="Protected" if snap.is_protected else "Unprotected",
              delta_color="normal" if snap.is_protected else "inverse")
with c2:
    st.metric("Estimated Bill", "PKR {:,.0f}".format(snap.bill["total"]),
              delta="{:.2f} PKR/unit".format(snap.bill["per_unit"]))
with c3:
    st.metric("Solar Generation", "{:.0f} kWh".format(snap.solar_monthly_kwh),
              delta="{:.0f} self-consumed".format(snap.solar_self_consumed))
with c4:
    st.metric("Battery Backup", "{:.0f} kWh".format(snap.system.battery_kwh),
              delta="~{:.0f} units shifted".format(snap.battery_shifted))

# =============================================================
# 11. THRESHOLD GAUGE
# =============================================================
st.markdown("### Protected Status Threshold (<={:.0f} units)".format(tariff.protected_limit))
col_g, col_i = st.columns([2, 1])

with col_g:
    gauge_max = max(300, tariff.protected_limit * 1.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=snap.grid_after_battery,
        number={"suffix": " units", "font": {"size": 40, "color": COLORS["primary"]}},
        delta={"reference": tariff.protected_limit,
               "increasing": {"color": COLORS["danger"]},
               "decreasing": {"color": COLORS["success"]}},
        gauge={
            "axis": {"range": [0, gauge_max], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, tariff.protected_limit * 0.75], "color": "#0D2818"},
                {"range": [tariff.protected_limit * 0.75, tariff.protected_limit],
                 "color": "#2D2800"},
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
        st.success("ON TRACK - {:.0f} units remaining".format(snap.remaining))
    elif snap.remaining > 0:
        st.warning("CAUTION - Only {:.0f} units left".format(snap.remaining))
    else:
        st.error("EXCEEDED by {:.0f} units".format(abs(snap.remaining)))

# =============================================================
# 12. SANKEY
# =============================================================
st.markdown("### Energy Flow")
fig_s = go.Figure(go.Sankey(
    node=dict(pad=20, thickness=25,
              line=dict(color=COLORS["border"], width=0.5),
              label=["Grid Import", "Solar PV", "Battery", "Home Load", "Grid Export"],
              color=[COLORS["primary"], COLORS["solar"], COLORS["battery"],
                     COLORS["danger"], COLORS["muted"]]),
    link=dict(
        source=[0, 1, 1, 2, 1], target=[3, 3, 2, 3, 4],
        value=[snap.grid_after_battery, snap.solar_self_consumed * 0.6,
               snap.solar_self_consumed * 0.4, snap.battery_shifted, snap.solar_export],
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
# 13. DONUT + HISTORICAL TREND
# =============================================================
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("### Appliance Breakdown")
    fig_d = go.Figure(go.Pie(
        labels=[a.name for a in snap.appliances],
        values=[a.monthly_kwh for a in snap.appliances],
        hole=0.55, marker=dict(colors=px.colors.qualitative.Bold),
        textinfo="label+percent", textfont=dict(color=COLORS["text"], size=11),
    ))
    fig_d.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": COLORS["text"]}, showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_d, use_container_width=True)

with c_right:
    st.markdown("### Previous Year Consumption Trend")

    hist_kwh = _parse_csv(st.session_state.hist_kwh)
    hist_bill = _parse_csv(st.session_state.hist_bill)

    if hist_kwh:
        if len(hist_kwh) <= 12:
            months = MONTH_LABELS[-len(hist_kwh):]
        else:
            months = ["M{}".format(i + 1) for i in range(len(hist_kwh))]

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=months, y=hist_kwh, mode="lines+markers", name="kWh",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=10, color=COLORS["primary"]),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
        ))
        if hist_bill and len(hist_bill) == len(hist_kwh):
            fig_t.add_trace(go.Scatter(
                x=months, y=hist_bill, mode="lines+markers", name="PKR bill",
                yaxis="y2",
                line=dict(color=COLORS["solar"], width=2, dash="dot"),
                marker=dict(size=8, color=COLORS["solar"]),
            ))
        fig_t.add_hline(y=tariff.protected_limit, line_dash="dash",
                        line_color=COLORS["danger"],
                        annotation_text="Protected Limit ({:.0f})".format(tariff.protected_limit))
        fig_t.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]},
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="#21262D", title="kWh"),
            yaxis2=dict(title="PKR", overlaying="y", side="right",
                        showgrid=False, tickfont=dict(color=COLORS["solar"])),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Enter previous-year monthly kWh in the sidebar to see trends.")

# =============================================================
# 14. ACTION CARDS
# =============================================================
st.markdown("---")
st.markdown("### Recommended Actions")

if snap.remaining > 20:
    st.markdown(
        '<div class="action-card">'
        '<b>You are on track!</b> Keep current consumption patterns. '
        'Your grid import is within the protected threshold.'
        '</div>',
        unsafe_allow_html=True,
    )
elif snap.remaining > 0:
    st.markdown(
        '<div class="action-card action-card-warning">'
        '<b>Caution: {:.0f} units remaining.</b> '
        'Take these actions immediately:'
        '</div>'.format(snap.remaining),
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Increase solar to reduce grid import - Each +1 kWp offsets ~110-150 units/month
    - Add battery capacity - Each +5 kWh shifts ~4-6 units/day from grid to solar
    - Shift AC/geyser usage to 10 AM-3 PM (solar peak)
    - Switch to LED bulbs - saves 3-5 units/month
    - Set AC to 26 degrees C - saves up to 24% AC energy
    """)
else:
    st.markdown(
        '<div class="action-card action-card-danger">'
        '<b>EXCEEDED by {:.0f} units!</b> '
        'Immediate action required:'
        '</div>'.format(abs(snap.remaining)),
        unsafe_allow_html=True,
    )
    st.markdown("""
    - Reduce consumption immediately - Unprotected rates are 2-3x higher
    - Add solar kWp urgently - Minimum 2 kWp additional recommended
    - Maximize battery - Store solar for night use
    - Do NOT install a second meter - LESCO crackdown active since 2026
    - Check separate-family exception (separate kitchen + wiring required)
    """)

st.markdown("---")
st.warning(
    "Legal Notice (September 2026): LESCO is actively cracking down on "
    "multiple meters used to artificially stay under 200 units. Households "
    "found doing this immediately lose protected status. Multiple meters are "
    "only permissible where separate families occupy distinct portions with "
    "independent kitchens and wiring."
)

# =============================================================
# 15. BILL BREAKDOWN
# =============================================================
st.markdown("### Bill Breakdown")
b = snap.bill
bc = st.columns(4)
with bc[0]:
    st.metric(
        "Protected ({} units)".format(b["p_units"]),
        "PKR {:,.0f}".format(b["p_cost"]),
        delta="@ {}/unit".format(tariff.protected_rate),
        delta_color="off",
    )
with bc[1]:
    if b["u_units"] > 0:
        st.metric(
            "Unprotected ({} units)".format(b["u_units"]),
            "PKR {:,.0f}".format(b["u_cost"]),
            delta="@ {}/unit".format(tariff.unprotected_rate),
            delta_color="off",
        )
    else:
        st.metric("Unprotected", "PKR 0", delta="none", delta_color="off")
with bc[2]:
    st.metric("Effective Rate", "PKR {:.2f}/unit".format(b["per_unit"]))
with bc[3]:
    st.metric("Total Bill", "PKR {:,.0f}".format(b["total"]))

if snap.solar_export > 0:
    st.info(
        "Net Billing Credit: Exported {:.0f} units at ~PKR {:.0f}/unit = "
        "PKR {:,.0f} credit (2026 Net Billing policy)".format(
            snap.solar_export, EXPORT_RATE, snap.export_credit
        )
    )

# =============================================================
# 16. AI INSIGHTS
# =============================================================
st.markdown("---")
st.markdown("### AI Energy Advisor (Groq / gpt-oss-120b)")

try:
    from agents import run_energy_analysis
    if st.button("Run AI Analysis", use_container_width=True):
        summary = ", ".join(
            "{} ({}W x {} x {}h)".format(a.name, a.watts, a.qty, a.hours)
            for a in snap.appliances
        )
        with st.spinner("Agents are analyzing your setup..."):
            st.session_state.ai_result = run_energy_analysis(
                appliance_data=summary,
                solar_kwp=snap.system.solar_kwp,
                battery_kwh=snap.system.battery_kwh,
                monthly_units=snap.grid_after_battery,
            )
    if st.session_state.ai_result:
        st.markdown(st.session_state.ai_result)
except ImportError:
    st.caption("AI advisor disabled - add agents.py to enable.")

# =============================================================
# 17. FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8B949E; font-size:0.8rem;'>"
    "Home Energy Manager v1.1 | Auto-calibrated tariff | "
    "Net Billing (NEPRA Feb 2026) | Streamlit + Plotly + CrewAI + Groq"
    "</p>",
    unsafe_allow_html=True,
)
