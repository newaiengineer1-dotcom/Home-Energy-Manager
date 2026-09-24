"""
CrewAI multi-agent system for Home Energy Management.
Model: Groq · openai/gpt-oss-120b

Requires GROQ_API_KEY in .streamlit/secrets.toml or environment.
"""
import os
from crewai import Agent, Task, Crew, LLM

GROQ_MODEL = "groq/openai/gpt-oss-120b"
REASONING_EFFORT = "medium"       # "low" | "medium" | "high"
TEMPERATURE = 0.7


def _get_llm() -> LLM | None:
    """Return a Groq LLM, or None if no API key."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return LLM(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        api_key=api_key,
    )


def _build_agents(llm: LLM) -> dict:
    """Construct the four specialized energy agents."""
    return {
        "forecaster": Agent(
            role="Consumption Forecaster",
            goal="Predict month-end units from appliance data and historical usage",
            backstory="Expert energy analyst specializing in Pakistani residential load profiles.",
            llm=llm, verbose=False,
        ),
        "solar_optimizer": Agent(
            role="Solar Optimizer",
            goal="Determine optimal kWp for self-consumption under Net Billing 2026",
            backstory="Solar engineer with expertise in Lahore irradiance (5.2 peak sun hours).",
            llm=llm, verbose=False,
        ),
        "battery_scheduler": Agent(
            role="Battery Scheduler",
            goal="Design charge/discharge schedule to maximize solar self-consumption",
            backstory="Battery storage specialist focused on lithium systems and TOU optimization.",
            llm=llm, verbose=False,
        ),
        "slab_guardian": Agent(
            role="Slab Guardian",
            goal="Monitor the protected-unit threshold and issue emergency alerts",
            backstory="LESCO tariff expert tracking protected-consumer regulations and NEPRA updates.",
            llm=llm, verbose=False,
        ),
    }


def run_energy_analysis(
    appliance_data: str,
    solar_kwp: float,
    battery_kwh: float,
    monthly_units: float,
) -> str:
    """Run the 4-agent crew and return the consolidated recommendation."""
    llm = _get_llm()
    if llm is None:
        return "⚠️ GROQ_API_KEY not configured. Add it to .streamlit/secrets.toml"

    agents = _build_agents(llm)

    tasks = [
        Task(
            description=f"Analyze appliances: {appliance_data}. Predict month-end units.",
            expected_output="Predicted monthly kWh with confidence range",
            agent=agents["forecaster"],
        ),
        Task(
            description=(
                f"With {solar_kwp} kWp solar and {monthly_units} units consumption, "
                f"recommend optimal solar size for Net Billing 2026."
            ),
            expected_output="Recommended kWp, expected self-consumption %, export credit",
            agent=agents["solar_optimizer"],
        ),
        Task(
            description=f"Design battery schedule for {battery_kwh} kWh to stay within the protected tier.",
            expected_output="Hour-by-hour charge/discharge schedule",
            agent=agents["battery_scheduler"],
        ),
        Task(
            description=f"Current grid import: {monthly_units} units. Assess protected status risk.",
            expected_output="Risk level, emergency actions, 6-month outlook",
            agent=agents["slab_guardian"],
        ),
    ]

    crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=False)
    return str(crew.kickoff())
