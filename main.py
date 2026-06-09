"""CLI entry point for TEYVALON: Nasha Town — Defend the Moon-Prayer Night."""

from ai_v2 import BayesianAIPlayer
from config import GameMode
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


def main() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║             TEYVALON: Nasha Town                 ║")
    print("║          Defend the Moon-Prayer Night            ║")
    print("╚══════════════════════════════════════════════════╝")

    num_players = _choose_int(
        "\nHow many players? (5-8, default 5): ", 5, 8, default=5
    )

    print("\nSelect game mode:")
    print("  1. Basic    — no special roles")
    print("  2. Prophecy — Columbina + Rerir (default)")
    print("  3. Veil     — Columbina, Lauma, Rerir, Dottore")
    modes = {1: GameMode.BASIC, 2: GameMode.PROPHECY, 3: GameMode.VEIL}
    mode_choice = _choose_int("> ", 1, 3, default=2)
    mode = modes[mode_choice]

    print("\nSelect AI version:")
    print("  1. v1 — rule-based (default)")
    print("  2. v2 — Bayesian probability model")
    ai_classes = {1: AIPlayer, 2: BayesianAIPlayer}
    ai_choice = _choose_int("> ", 1, 2, default=1)
    ai_class = ai_classes[ai_choice]

    game = Game(num_players=num_players, mode=mode, ai_class=ai_class)
    game.run()

    print("\nThanks for playing TEYVALON!\n")


if __name__ == "__main__":
    main()
