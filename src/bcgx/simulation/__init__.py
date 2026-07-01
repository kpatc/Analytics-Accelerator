"""Simulation package — scenario simulation engine and financial impact calculator."""

from bcgx.simulation.engine import (
    ScenarioInput,
    ScenarioOutput,
    ScenarioType,
    SimulationEngine,
)
from bcgx.simulation.impact import FinancialImpact, calculate_financial_impact
from bcgx.simulation.scenarios import PRESET_SCENARIOS

__all__ = [
    "ScenarioInput",
    "ScenarioOutput",
    "ScenarioType",
    "SimulationEngine",
    "FinancialImpact",
    "calculate_financial_impact",
    "PRESET_SCENARIOS",
]
