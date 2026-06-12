"""CLI entry point for TEYVALON: Nasha Town — Defend the Moon-Prayer Night."""

import random
import sys
import time

from ai_v2 import BayesianAIPlayer
from ai_v3 import JointBayesianAIPlayer
from config import MODE_SPECIAL_ROLES, Camp, GameMode, Role
from game import Game
from player import AIPlayer


def _choose_int(prompt: str, low: int, high: int, default: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if low <= val <= high:
                return val
            print(f"Enter a number between {low} and {high}.")
        except ValueError:
            print("Invalid input.")


def _available_roles(mode: GameMode) -> list[Role]:
    """Return all roles available in a given mode."""
    specials = MODE_SPECIAL_ROLES[mode]
    roles = list(specials[Camp.TOWNSFOLK]) + list(specials[Camp.ABYSSAL])
    roles += [Role.TOWNSFOLK, Role.ABYSSAL]
    return roles


def main() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║             TEYVALON: Nasha Town                 ║")
    print("║          Defend the Moon-Prayer Night            ║")
    print("╚══════════════════════════════════════════════════╝")

    # Seed
    default_seed = int(time.time() * 1000) % 1_000_000
    raw = input(f"\nRandom seed (default {default_seed}): ").strip()
    seed = int(raw) if raw.isdigit() else default_seed
    random.seed(seed)
    print(f"Seed: {seed}")

    num_players = _choose_int(
        "\nHow many players? (5-8, default 5): ", 5, 8, default=5
    )

    print("\nSelect game mode:")
    print("  1. Basic    — no special roles")
    print("  2. Prophecy — Columbina + Rerir")
    print("  3. Veil     — Columbina, Lauma, Rerir, Dottore (default)")
    modes = {1: GameMode.BASIC, 2: GameMode.PROPHECY, 3: GameMode.VEIL}
    mode_choice = _choose_int("> ", 1, 3, default=3)
    mode = modes[mode_choice]

    print("\nSelect AI version:")
    print("  1. v1 — rule-based")
    print("  2. v2 — Bayesian (mean-field)")
    print("  3. v3 — Bayesian (joint distribution, default)")
    ai_classes = {1: AIPlayer, 2: BayesianAIPlayer, 3: JointBayesianAIPlayer}
    ai_choice = _choose_int("> ", 1, 3, default=3)
    ai_class = ai_classes[ai_choice]

    max_proposals = _choose_int(
        "\nMax proposals per wave? (3-5, default 3): ", 3, 5, default=3
    )

    # Role selection
    roles = _available_roles(mode)
    print("\nSelect your role:")
    print("  0. Random (default)")
    for i, r in enumerate(roles):
        camp_tag = "Town" if r.camp == Camp.TOWNSFOLK else "Abyss"
        print(f"  {i + 1}. {r.display_name} ({camp_tag})")
    role_choice = _choose_int("> ", 0, len(roles), default=0)
    human_role = roles[role_choice - 1] if role_choice > 0 else None

    belief_panel = "--no-panel" not in sys.argv
    game = Game(
        num_players=num_players,
        mode=mode,
        max_proposals=max_proposals,
        ai_class=ai_class,
        human_role=human_role,
        belief_panel=belief_panel,
    )
    game.run()

    print("\nThanks for playing TEYVALON!\n")


if __name__ == "__main__":
    main()
