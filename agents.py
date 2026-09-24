"""
CrewAI multi-agent system for Home Energy Management.
Model: Groq / openai/gpt-oss-120b
Includes retry-with-backoff and reduced agent count to stay within
Groq free-tier limits (1,000 RPD, 8,000 TPM).
"""
import os
import time
import logging
from typing import Optional

from crewai import Agent, Task, Crew, LLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
GROQ_MODEL = "groq/openai/gpt-oss-120b"
REASONING_EFFORT = "low"          # "low" uses fewer tokens → fewer rate limits
TEMPERATURE = 0.7
MAX_ATTEMPTS = 3                  # retry up to 3 times
MAX_RPM = 20                      # hard cap on requests/min per agent

# Quiet down litellm/crewai logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("energy_agents")


# ─────────────────────────────────────────────────────────────
# RETRY WRAPPER
# ─────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _kickoff_with_retry(crew: Crew):
    """Run crew.kickoff() with exponential backoff on any error."""
    return crew.kickoff()


# ─────────────────────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────────────────────
def _get_llm() -> Optional[LLM]:
    """Return a Groq LLM, or None if no API key."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return LLM(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        api_key=api_key,
        max_retries=0,            # we handle retries ourselves
    )


# ─────────────────────────────────────────────────────────────
# AGENT BUILDER (2 agents instead of 4 → fewer API calls)
# ─────────────────────────────────────────────────────────────
def _build_agents(llm: LLM) -> dict:
    """Two combined agents to halve the number of LLM calls."""
    return {
        "analyst": Agent(
            role="Energy Analyst",
            goal=(
                "Forecast month-end consumption and assess protected-status risk "
                "under the LESCO 200-unit threshold."
            ),
            backstory=(
                "Expert energy analyst specializing in Pakistani residential "
                "load profiles, LESCO tariffs, and protected-consumer rules."
            ),
            llm=llm,
            verbose=False,
            max_rpm=MAX_RPM,
        ),
        "optimizer": Agent(
            role="Solar and Battery Optimizer",
            goal=(
                "Recommend solar kWp and battery kWh to keep grid import "
                "within the protected tier."
            ),
            backstory=(
                "Solar and battery engineer with expertise in Lahore irradiance "
                "(5.2 peak sun hours) and Net Billing 2026."
            ),
            llm=llm,
            verbose=False,
            max_rpm=MAX_RPM,
        ),
    }


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def run_energy_analysis(
    appliance_data: str,
    solar_kwp: float,
    battery_kwh: float,
    monthly_units: float,
) -> str:
    """
    Run a 2-agent crew and return the consolidated recommendation.
    Retries automatically on rate-limit errors.
    """
    llm = _get_llm()
    if llm is None:
        return "⚠️ GROQ_API_KEY not configured. Add it to .streamlit/secrets.toml"

    agents = _build_agents(llm)

    tasks = [
        Task(
            description=(
                f"Analyze these appliances: {appliance_data}. "
                f"Predict month-end units and assess whether the household "
                f"is at risk of exceeding the protected threshold. "
                f"Current grid import: {monthly_units} units."
            ),
            expected_output=(
                "A short report with: (1) predicted month-end units, "
                "(2) risk level (Low/Medium/High), (3) top 3 actions."
            ),
            agent=agents["analyst"],
        ),
        Task(
            description=(
                f"Given {solar_kwp} kWp solar and {battery_kwh} kWh battery, "
                f"recommend the optimal system size to keep grid import "
                f"under the protected threshold. Consider Net Billing 2026 "
                f"and Lahore irradiance."
            ),
            expected_output=(
                "A short recommendation with: (1) recommended kWp, "
                "(2) recommended battery kWh, (3) expected monthly saving in PKR."
            ),
            agent=agents["optimizer"],
        ),
    ]

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=False,
        max_rpm=MAX_RPM,
    )

    try:
        result = _kickoff_with_retry(crew)
        return str(result)
    except Exception as e:
        err = str(e)
        if "RateLimitError" in err or "429" in err:
            return (
                "⚠️ **Groq rate limit reached.**\n\n"
                "The free tier allows 1,000 requests/day and 8,000 tokens/min "
                "for `gpt-oss-120b`. Wait about 60 seconds and try again, or:\n\n"
                "- Reduce the number of appliances you're analyzing\n"
                "- Use a smaller model (see README)\n"
                "- Upgrade to a Groq paid plan at https://console.groq.com/settings/billing"
            )
        return f"⚠️ AI analysis failed: {err}"
