"""Benchmark scenarios package."""

from arm_opt.scenarios.base_scenario import BaseScenario
from arm_opt.scenarios.rest_to_rest import RestToRestScenario
from arm_opt.scenarios.obstacle import ObstacleScenario
from arm_opt.scenarios.high_speed import HighSpeedScenario
from arm_opt.scenarios.under_torqued import UnderTorquedScenario
from arm_opt.scenarios.reversal import ReversalScenario
from arm_opt.scenarios.swing_up import SwingUpScenario

SCENARIOS = {
    "rest_to_rest": RestToRestScenario,
    "obstacle": ObstacleScenario,
    "high_speed": HighSpeedScenario,
    "under_torqued": UnderTorquedScenario,
    "reversal": ReversalScenario,
    "swing_up": SwingUpScenario,
}

__all__ = [
    "BaseScenario",
    "RestToRestScenario",
    "ObstacleScenario",
    "HighSpeedScenario",
    "UnderTorquedScenario",
    "ReversalScenario",
    "SwingUpScenario",
    "SCENARIOS",
]