"""Balance benchmark: run batch AI-vs-AI games and print win-rate stats."""

import io
import contextlib
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from config import GameMode, CAMP_SIZES, DEFAULT_MAX_PROPOSALS
from game import Game
from ai_v2 import BayesianAIPlayer
from ai_v3 import JointBayesianAIPlayer
from player import AIPlayer

CONFIGS = [
    (5, GameMode.BASIC, "Basic 5p"),
    (6, GameMode.BASIC, "Basic 6p"),
    (7, GameMode.BASIC, "Basic 7p"),
    (8, GameMode.BASIC, "Basic 8p"),
    (5, GameMode.PROPHECY, "Prophecy 5p"),
    (6, GameMode.PROPHECY, "Prophecy 6p"),
    (7, GameMode.PROPHECY, "Prophecy 7p"),
    (8, GameMode.PROPHECY, "Prophecy 8p"),
    (5, GameMode.VEIL, "Veil 5p"),
    (6, GameMode.VEIL, "Veil 6p"),
    (7, GameMode.VEIL, "Veil 7p"),
    (8, GameMode.VEIL, "Veil 8p"),
]

AI_CLASSES = {"v1": AIPlayer, "v2": BayesianAIPlayer, "v3": JointBayesianAIPlayer}


@dataclass
class ConfigResult:
    label: str
    town_wins: int = 0
    abyss_wins: int = 0
    hunt_triggered: int = 0
    hunt_success: int = 0
    total: int = 0
    # How many evil were deterministically identified by townsfolk
    deduced_counts: list = None  # list of ints: how many evil deduced per game

    def __post_init__(self):
        self.deduced_counts = []


def _run_single(args: tuple) -> str:
    """Run one game and return outcome string. Runs in worker process."""
    num_players, mode, seed, ai_version, max_proposals = args
    ai_class = AI_CLASSES[ai_version]

    random.seed(seed)
    g = Game(num_players=num_players, mode=mode, max_proposals=max_proposals, ai_class=ai_class, belief_panel=False)

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
        outcome = "hunt_found"
    elif "Rerir guessed wrong" in output:
        outcome = "hunt_wrong"
    elif "Abyssals win" in output or "All proposals rejected" in output:
        outcome = "abyss"
    else:
        outcome = "town"

    # Count how many evil each townsfolk has deterministically identified
    # (intersection of all surviving hypotheses = certain evil)
    from config import Camp
    deduced = 0
    town_count = 0
    for p in g.players:
        if p.camp != Camp.TOWNSFOLK or not hasattr(p, '_joint') or not p._joint:
            continue
        town_count += 1
        certain_evil = set.intersection(*[set(s) for s in p._joint])
        deduced += len(certain_evil)
    avg_deduced = deduced / max(town_count, 1)

    return outcome, avg_deduced


def run_batch(ai_version: str, num_games: int = 500, workers: int = 0, filter_str: str = "", max_proposals: int = DEFAULT_MAX_PROPOSALS) -> None:
    if workers <= 0:
        workers = os.cpu_count() or 4

    ai_class_name = AI_CLASSES[ai_version].__name__
    print(f"Running {num_games} games × {len(CONFIGS)} configs with {ai_class_name} (max_proposals={max_proposals})")
    print(f"Using {workers} worker processes\n")
    print(f"{'Config':15s} {'Town':>6s} {'← balance →':^22} {'Abyss':<6s} | {'Hunts':>9s} {'Acc':>4s}{'':>12s} {'Deduced':>7s}")
    print("-" * 89)

    # Build all tasks (filter by label substring if given)
    all_tasks: list[tuple[tuple, str]] = []
    for num_players, mode, label in CONFIGS:
        if filter_str and filter_str.lower() not in label.lower():
            continue
        for seed in range(1, num_games + 1):
            all_tasks.append(((num_players, mode, seed, ai_version, max_proposals), label))

    # Run all games in parallel
    configs = [(np, m, l) for np, m, l in CONFIGS if not filter_str or filter_str.lower() in l.lower()]
    results: dict[str, ConfigResult] = {}
    for _, _, label in configs:
        results[label] = ConfigResult(label=label)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        task_args = [t[0] for t in all_tasks]
        task_labels = [t[1] for t in all_tasks]

        for label, result in zip(task_labels, pool.map(_run_single, task_args, chunksize=50)):
            outcome, avg_deduced = result
            r = results[label]
            r.total += 1
            r.deduced_counts.append(avg_deduced)
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

    # Color helpers
    G = "\033[92m"  # green
    R = "\033[91m"  # red
    Y = "\033[93m"  # yellow
    D = "\033[2m"   # dim
    B = "\033[1m"   # bold
    X = "\033[0m"   # reset

    # Print results
    for _, _, label in configs:
        r = results[label]
        n = r.total
        t_pct_val = r.town_wins / n * 100
        a_pct_val = r.abyss_wins / n * 100
        t_pct = f"{t_pct_val:.1f}%"
        a_pct = f"{a_pct_val:.1f}%"
        h_str = f"{r.hunt_triggered}/{n}"
        h_acc_val = r.hunt_success / max(r.hunt_triggered, 1) * 100 if r.hunt_triggered else 0
        h_acc = f"{h_acc_val:.0f}%" if r.hunt_triggered else "—"

        # Win-rate bar: 20 chars wide, green for town, red for abyss
        bar_w = 22
        t_bar = round(t_pct_val / 100 * bar_w)
        a_bar = bar_w - t_bar
        bar = f"{G}{'█' * t_bar}{R}{'█' * a_bar}{X}"

        # Color the winner side
        if t_pct_val > 55:
            t_col, a_col = G, D
        elif a_pct_val > 55:
            t_col, a_col = D, R
        else:
            t_col, a_col = Y, Y  # balanced

        # Hunt accuracy bar: 10 chars, red=success(abyss wins), green=fail(town wins)
        if r.hunt_triggered:
            h_bar_w = 10
            h_fill = round(h_acc_val / 100 * h_bar_w)
            h_empty = h_bar_w - h_fill
            h_bar = f"{R}{'█' * h_fill}{G}{'█' * h_empty}{X}"
            h_display = f"{h_acc:>4s} {h_bar}"
        else:
            h_display = f"{'—':>4s} {D}{'·' * 10}{X}"

        # Deduced evil stats
        avg_ded = sum(r.deduced_counts) / max(len(r.deduced_counts), 1)
        ded_str = f"{avg_ded:.2f}"

        print(
            f"{label:15s} {t_col}{t_pct:>6s}{X} {bar} {a_col}{a_pct:<6s}{X}"
            f" | {h_str:>9s} {h_display}  {ded_str}"
        )


def main() -> None:
    num_games = 500
    ai_version = "v3"
    workers = 0  # auto

    max_proposals = DEFAULT_MAX_PROPOSALS
    filter_str = ""
    args = sys.argv[1:]
    for arg in args:
        if arg.isdigit():
            num_games = int(arg)
        elif arg in ("v1", "v2", "v3"):
            ai_version = arg
        elif arg.startswith("-j"):
            workers = int(arg[2:])
        elif arg.startswith("-p"):
            max_proposals = int(arg[2:])
        else:
            filter_str = arg

    run_batch(ai_version, num_games, workers, filter_str, max_proposals)


if __name__ == "__main__":
    main()
