"""Game configuration, constants, and role definitions."""

from enum import Enum


class Camp(Enum):
    TOWNSFOLK = "Nasha Townsfolk"
    ABYSSAL = "Abyssals"


class Role(Enum):
    # Townsfolk roles
    TOWNSFOLK = ("Townsfolk", Camp.TOWNSFOLK)
    COLUMBINA = ("Columbina", Camp.TOWNSFOLK)  # Merlin — knows all Abyssals
    LAUMA = ("Lauma", Camp.TOWNSFOLK)  # Percival — sees Columbina & Dottore

    # Abyssal roles
    ABYSSAL = ("Abyssal", Camp.ABYSSAL)
    RERIR = ("Rerir", Camp.ABYSSAL)  # Assassin — hunts Columbina
    DOTTORE = ("Dottore", Camp.ABYSSAL)  # Morgana — fake Moon to Lauma

    def __init__(self, display_name: str, camp: Camp):
        self.display_name = display_name
        self.camp = camp


class GameMode(Enum):
    BASIC = "Basic"
    PROPHECY = "Prophecy"
    VEIL = "Veil"


# (townsfolk_count, abyssal_count)
CAMP_SIZES: dict[int, tuple[int, int]] = {
    5: (3, 2),
    6: (4, 2),
    7: (4, 3),
    8: (5, 3),
}

# Team size for each wave, indexed by total player count
WAVE_TEAM_SIZES: dict[int, list[int]] = {
    5: [2, 3, 2, 3, 3],
    6: [2, 3, 4, 3, 4],
    7: [2, 3, 3, 4, 4],
    8: [3, 4, 4, 5, 5],
}

# Wave 4 (index 3) requires 2+ fake bombs to fail when player count >= this
WAVE4_DOUBLE_FAIL_THRESHOLD = 7

# Default max team proposals per wave before Abyssals auto-win
DEFAULT_MAX_PROPOSALS = 3

# Special roles to assign per game mode
MODE_SPECIAL_ROLES: dict[GameMode, dict[Camp, list[Role]]] = {
    GameMode.BASIC: {
        Camp.TOWNSFOLK: [],
        Camp.ABYSSAL: [],
    },
    GameMode.PROPHECY: {
        Camp.TOWNSFOLK: [Role.COLUMBINA],
        Camp.ABYSSAL: [Role.RERIR],
    },
    GameMode.VEIL: {
        Camp.TOWNSFOLK: [Role.COLUMBINA, Role.LAUMA],
        Camp.ABYSSAL: [Role.RERIR, Role.DOTTORE],
    },
}

# AI player names
AI_NAMES = [
    "Pulonia",
    "Chard Lombro",
    "CYJ Ducky",
    "Knuckle Duckle",
    "70% Sweet",
    "Spy-Bot No.1",
    "Luonnotar",
    "Maalaus",
]
