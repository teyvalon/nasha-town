"""Balance benchmark: run batch AI-vs-AI games and print win-rate stats."""

import io
import contextlib
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

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

AI_CLASSES = {"v1": AIPlayer, "v2": BayesianAIPlayer}


@dataclass
class ConfigResult:
    label: str
    town_wins: int = 0
    abyss_wins: int = 0
    hunt_triggered: int = 0
    hunt_success: int = 0
    total: int = 0


def _run_single(args: tuple) -> str:
    """Run one game and return outcome string. Runs in worker process."""
    num_players, mode, seed, ai_version = args
    ai_class = AI_CLASSES[ai_version]

    random.seed(seed)
    g = Game(num_players=num_players, mode=mode, ai_class=ai_class)

    # Patch setup to use all-AI players
    orig_setup = g.setup

    def all_ai_setup():
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

    g.setup = all_ai_setup

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g.run()
    output = buf.getvalue()

    if "Rerir found" in output:
        return "hunt_found"
    elif "Rerir guessed wrong" in output:
        return "hunt_wrong"
    elif "Abyssals win" in output or "All proposals rejected" in output:
        return "abyss"
    else:
        return "town"


def run_batch(ai_version: str, num_games: int = 500, workers: int = 0) -> None:
    if workers <= 0:
        workers = os.cpu_count() or 4

    ai_class_name = AI_CLASSES[ai_version].__name__
    print(f"Running {num_games} games × {len(CONFIGS)} configs with {ai_class_name}")
    print(f"Using {workers} worker processes\n")
    print(f"{'Config':15s} | {'Town':>6s}  {'Abyss':>6s} | {'Hunt':>9s}  {'Acc':>4s}")
    print("-" * 58)

    # Build all tasks
    all_tasks: list[tuple[tuple, str]] = []  # (args, label)
    for num_players, mode, label in CONFIGS:
        for seed in range(1, num_games + 1):
            all_tasks.append(((num_players, mode, seed, ai_version), label))

    # Run all games in parallel
    results: dict[str, ConfigResult] = {}
    for _, _, label in CONFIGS:
        results[label] = ConfigResult(label=label)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        task_args = [t[0] for t in all_tasks]
        task_labels = [t[1] for t in all_tasks]

        for label, outcome in zip(task_labels, pool.map(_run_single, task_args, chunksize=50)):
            r = results[label]
            r.total += 1
            if outcome == "hunt_found":
                r.hunt_triggered += 1
                r.hunt_success += 1
                r.abyss_wins += 1
            elif outcome == "hunt_wrong":
                r.hunt_triggered += 1
                r.town_wins += 1
            elif outcome == "abyss":
                r.abyss_wins += 1
            else:
                r.town_wins += 1

    # Print results
    for _, _, label in CONFIGS:
        r = results[label]
        n = r.total
        t_pct = f"{r.town_wins / n * 100:.1f}%"
        a_pct = f"{r.abyss_wins / n * 100:.1f}%"
        h_str = f"{r.hunt_triggered}/{n}"
        h_acc = f"{r.hunt_success / max(r.hunt_triggered, 1) * 100:.0f}%" if r.hunt_triggered else "—"
        print(f"{label:15s} | {t_pct:>6s}  {a_pct:>6s} | {h_str:>9s}  {h_acc:>4s}")


def main() -> None:
    num_games = 500
    ai_version = "v2"
    workers = 0  # auto

    args = sys.argv[1:]
    for arg in args:
        if arg.isdigit():
            num_games = int(arg)
        elif arg in ("v1", "v2"):
            ai_version = arg
        elif arg.startswith("-j"):
            workers = int(arg[2:])

    run_batch(ai_version, num_games, workers)


if __name__ == "__main__":
    main()
