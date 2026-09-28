from arm_opt.scenarios.base_scenario import BaseScenario
from arm_opt.scenarios.rest_to_rest import RestToRestScenario
from arm_opt.scenarios.obstacle import ObstacleScenario
from arm_opt.scenarios.high_speed import HighSpeedScenario
from arm_opt.scenarios.swing_up import SwingUpScenario

SCENARIOS = {
    "rest_to_rest": RestToRestScenario,
    "swing_up": SwingUpScenario,
    "obstacle": ObstacleScenario,
    "high_speed": HighSpeedScenario,
}

__all__ = [
    "BaseScenario",
    "RestToRestScenario",
    "SwingUpScenario",
    "ObstacleScenario",
    "HighSpeedScenario",
    "SCENARIOS",
]
