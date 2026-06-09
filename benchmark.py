"""Balance benchmark: run batch AI-vs-AI games and print win-rate stats."""

import io
import contextlib
import random
import sys

from config import GameMode, CAMP_SIZES
from game import Game
from ai_v2 import BayesianAIPlayer
from player import AIPlayer

CONFIGS = [
    (5, GameMode.BASIC, "Basic 5p"),
    (5, GameMode.PROPHECY, "Prophecy 5p"),
    (5, GameMode.VEIL, "Veil 5p"),
    (6, GameMode.PROPHECY, "Prophecy 6p"),
    (6, GameMode.VEIL, "Veil 6p"),
    (7, GameMode.PROPHECY, "Prophecy 7p"),
    (7, GameMode.VEIL, "Veil 7p"),
    (8, GameMode.PROPHECY, "Prophecy 8p"),
    (8, GameMode.VEIL, "Veil 8p"),
]


def run_batch(ai_class: type, num_games: int = 500) -> None:
    print(f"Running {num_games} games per config with {ai_class.__name__}\n")
    print(f"{'Config':15s} | {'Town':>6s}  {'Abyss':>6s} | {'Hunt':>9s}  {'Acc':>4s}")
    print("-" * 58)

    for num_players, mode, label in CONFIGS:
        town_wins = 0
        abyss_wins = 0
        hunt_triggered = 0
        hunt_success = 0

        for seed in range(1, num_games + 1):
            random.seed(seed)
            g = Game(num_players=num_players, mode=mode, ai_class=ai_class)

            # Replace human player with AI
            original_setup = g.setup

            def patched_setup(g=g, seed=seed):
                random.seed(seed)
                roles = g._make_roles()
                random.shuffle(roles)
                g.players = []
                for i in range(g.num_players):
                    g.players.append(ai_class(f"P{i}", roles[i]))
                g.leader_idx = random.randrange(g.num_players)
                g._night_phase()
                num_evil = CAMP_SIZES[g.num_players][1]
                for p in g.players:
                    p.init_beliefs(g.players, num_evil)

            g.setup = patched_setup

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                g.run()
            output = buf.getvalue()

            if "Rerir found" in output:
                hunt_triggered += 1
                hunt_success += 1
                abyss_wins += 1
            elif "Rerir guessed wrong" in output:
                hunt_triggered += 1
                town_wins += 1
            elif "Abyssals win" in output or "All proposals rejected" in output:
                abyss_wins += 1
            else:
                town_wins += 1

        t_pct = f"{town_wins / num_games * 100:.1f}%"
        a_pct = f"{abyss_wins / num_games * 100:.1f}%"
        h_str = f"{hunt_triggered}/{num_games}"
        h_acc = f"{hunt_success / max(hunt_triggered, 1) * 100:.0f}%" if hunt_triggered else "—"
        print(f"{label:15s} | {t_pct:>6s}  {a_pct:>6s} | {h_str:>9s}  {h_acc:>4s}")


def main() -> None:
    num_games = 500
    ai_version = "v2"

    args = sys.argv[1:]
    for arg in args:
        if arg.isdigit():
            num_games = int(arg)
        elif arg in ("v1", "v2"):
            ai_version = arg

    ai_class = BayesianAIPlayer if ai_version == "v2" else AIPlayer
    run_batch(ai_class, num_games)


if __name__ == "__main__":
    main()
