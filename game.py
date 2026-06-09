"""Game engine: orchestrates the full TEYVALON game flow."""

from __future__ import annotations

import random

from config import (
    AI_NAMES,
    CAMP_SIZES,
    DEFAULT_MAX_PROPOSALS,
    MODE_SPECIAL_ROLES,
    WAVE4_DOUBLE_FAIL_THRESHOLD,
    WAVE_TEAM_SIZES,
    Camp,
    GameMode,
    Role,
)
from player import AIPlayer, HumanPlayer, Player

# ANSI color helpers
CYAN = "\033[96m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class Game:
    def __init__(
        self,
        num_players: int = 5,
        mode: GameMode = GameMode.PROPHECY,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        ai_class: type[Player] = AIPlayer,
    ):
        assert num_players in CAMP_SIZES, f"Unsupported player count: {num_players}"
        self.num_players = num_players
        self.mode = mode
        self.max_proposals = max_proposals
        self.ai_class = ai_class
        self.players: list[Player] = []
        self.wave_sizes = WAVE_TEAM_SIZES[num_players]
        self.town_score = 0
        self.abyssal_score = 0
        self.leader_idx = 0

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        roles = self._make_roles()
        random.shuffle(roles)

        # First player is always the human
        self.players.append(HumanPlayer("You", roles[0]))
        for i in range(1, self.num_players):
            self.players.append(self.ai_class(AI_NAMES[i - 1], roles[i]))

        self.leader_idx = random.randrange(self.num_players)
        self._night_phase()

        # Initialize belief models (no-op for v1 AI)
        num_evil = CAMP_SIZES[self.num_players][1]
        for p in self.players:
            p.init_beliefs(self.players, num_evil)

    def _make_roles(self) -> list[Role]:
        n_town, n_abyssal = CAMP_SIZES[self.num_players]
        specials = MODE_SPECIAL_ROLES[self.mode]

        town_roles = list(specials[Camp.TOWNSFOLK])
        abyssal_roles = list(specials[Camp.ABYSSAL])
        town_roles += [Role.TOWNSFOLK] * (n_town - len(town_roles))
        abyssal_roles += [Role.ABYSSAL] * (n_abyssal - len(abyssal_roles))
        return town_roles + abyssal_roles

    def _night_phase(self) -> None:
        """Reveal secret information based on roles."""
        abyssals = [p for p in self.players if p.camp == Camp.ABYSSAL]

        for p in self.players:
            if p.camp == Camp.ABYSSAL:
                p.known_info["abyssal_allies"] = [a for a in abyssals if a is not p]

            if p.role == Role.COLUMBINA:
                p.known_info["known_abyssals"] = list(abyssals)

            if p.role == Role.LAUMA:
                moon_players = [
                    x for x in self.players if x.role in (Role.COLUMBINA, Role.DOTTORE)
                ]
                random.shuffle(moon_players)
                p.known_info["moon_power_players"] = moon_players

    # ── Display helpers ───────────────────────────────────────────────

    def _show_role_info(self) -> None:
        human = self.players[0]
        print(f"\n{'=' * 50}")
        print(f"  Your role: {human.role.display_name} ({human.camp.value})")

        if "abyssal_allies" in human.known_info:
            names = ", ".join(p.name for p in human.known_info["abyssal_allies"])
            print(f"  Abyssal ally: {names}")

        if "known_abyssals" in human.known_info:
            names = ", ".join(p.name for p in human.known_info["known_abyssals"])
            print(f"  You see the Abyssals: {names}")

        if "moon_power_players" in human.known_info:
            names = ", ".join(p.name for p in human.known_info["moon_power_players"])
            print(f"  Players with Moon's power: {names}")
            print("  (One is Columbina, one is Dottore — you don't know which)")

        print(f"{'=' * 50}")

    def _show_scoreboard(self, wave: int) -> None:
        print(f"\n{'#' * 50}")
        print(f"  Wave {wave + 1}/5 — {self.wave_sizes[wave]} Wild Hunts incoming!")
        print(f"  Score: {GREEN}Townsfolk {self.town_score}{RESET} - {RED}Abyssals {self.abyssal_score}{RESET}")
        print(f"{'#' * 50}")

    def _reveal_roles(self, winner: Camp | None = None) -> None:
        print(f"\n{'=' * 50}")
        print("  ROLES REVEALED")
        print(f"{'=' * 50}")
        for p in self.players:
            color = GREEN if p.camp == Camp.TOWNSFOLK else RED
            print(f"  {p.name}: {color}{p.role.display_name} ({p.camp.value}){RESET}")
        if winner is not None:
            human_camp = self.players[0].camp
            if human_camp == winner:
                print(f"\n  {BOLD}{GREEN}*** YOU WIN! ***{RESET}")
            else:
                print(f"\n  {BOLD}{RED}*** YOU LOSE ***{RESET}")

    # ── Main game loop ────────────────────────────────────────────────

    def run(self) -> None:
        self.setup()

        print(f"\nGame Mode: {self.mode.value} | Players: {self.num_players}")
        player_list = ", ".join(p.name for p in self.players)
        print(f"Seats: {player_list}")
        n_town = sum(1 for p in self.players if p.camp == Camp.TOWNSFOLK)
        n_abyss = self.num_players - n_town
        print(f"Camps: {n_town} Townsfolk vs {n_abyss} Abyssals")

        self._show_role_info()

        for wave in range(5):
            self._show_scoreboard(wave)
            result = self._play_wave(wave)

            if result is None:
                print(f"\n{RED}All proposals rejected! The Abyssals win!{RESET}")
                self._reveal_roles(Camp.ABYSSAL)
                return

            if result:
                self.town_score += 1
                print(f"\n{GREEN}>>> Wave {wave + 1}: Townsfolk defended successfully!{RESET}")
            else:
                self.abyssal_score += 1
                print(f"\n{RED}>>> Wave {wave + 1}: Wild Hunts broke through!{RESET}")

            if self.town_score >= 3:
                return self._townsfolk_win_check()
            if self.abyssal_score >= 3:
                print(f"\n{BOLD}{RED}The Abyssals win the game!{RESET}")
                self._reveal_roles(Camp.ABYSSAL)
                return

        # Tie-breaker shouldn't happen with best-of-5, but just in case
        print("\nGame over!")
        self._reveal_roles()

    # ── Wave (round) logic ────────────────────────────────────────────

    def _play_wave(self, wave: int) -> bool | None:
        """Run proposals and mission for one wave.

        Returns True if townsfolk win the wave, False if abyssals win,
        or None if all proposals were rejected (abyssals auto-win).
        """
        team_size = self.wave_sizes[wave]

        for proposal in range(self.max_proposals):
            leader = self.players[self.leader_idx]
            print(
                f"\n--- Proposal {proposal + 1}/{self.max_proposals}"
                f" | Leader: {leader.name} ---"
            )

            team = leader.propose_team(self.players, team_size, wave)
            team_names = ", ".join(f"{BOLD}{p.name}{RESET}" for p in team)
            print(f"Proposed team: [{team_names}]")

            approved = self._vote(team, wave, proposal)
            if approved:
                return self._execute_mission(team, wave)

            # Rejected — rotate leader
            self.leader_idx = (self.leader_idx + 1) % self.num_players

        return None

    def _vote(self, team: list[Player], wave: int, proposal: int) -> bool:
        """Everyone votes. Returns True if majority approves."""
        votes: dict[Player, bool] = {}
        for p in self.players:

            votes[p] = p.vote(team, wave, proposal, self.max_proposals)

        approve = sum(1 for v in votes.values() if v)
        reject = self.num_players - approve

        print("\nVotes: ", end="")
        for p, v in votes.items():
            if v:
                print(f"{CYAN}{p.name}:Y{RESET}  ", end="")
            else:
                print(f"{RED}{p.name}:N{RESET}  ", end="")
        print(f"\nResult: {CYAN}{approve} approve{RESET} / {RED}{reject} reject{RESET}", end="")

        # Notify all players of the vote outcome
        for p in self.players:
            p.observe_team_vote(team, votes, wave)

        if approve > self.num_players / 2:
            print(f" — {GREEN}Approved!{RESET}")
            return True
        else:
            print(f" — {RED}Rejected!{RESET}")
            return False

    def _execute_mission(self, team: list[Player], wave: int) -> bool:
        """Team places bombs. Returns True if townsfolk win this wave."""
        print("\nTeam enters the abyss mist...")

        bombs: list[bool] = []
        for p in team:
            bombs.append(p.place_bomb(wave, team))

        fake_count = bombs.count(False)

        # Wave 4 (index 3) with 7+ players needs 2 fakes to fail
        fails_needed = 1
        if wave == 3 and self.num_players >= WAVE4_DOUBLE_FAIL_THRESHOLD:
            fails_needed = 2
            print("(Capitano sent reinforcements — need 2+ fakes for Abyssals to score)")

        real_count = bombs.count(True)
        real_str = f"{GREEN}{real_count} real{RESET}"
        fake_str = f"{RED}{fake_count} fake{RESET}" if fake_count else f"{DIM}0 fake{RESET}"
        print(f"\nBombs revealed: {real_str}, {fake_str}")

        # Notify all players of mission result
        for p in self.players:
            p.observe_mission_result(team, fake_count, wave)

        # Rotate leader after mission
        self.leader_idx = (self.leader_idx + 1) % self.num_players

        return fake_count < fails_needed

    # ── End-game ──────────────────────────────────────────────────────

    def _townsfolk_win_check(self) -> None:
        """Handle townsfolk reaching 3 wins — Rerir may hunt Columbina."""
        has_rerir = any(p.role == Role.RERIR for p in self.players)
        if not has_rerir:
            print(f"\n{BOLD}{GREEN}The Townsfolk win the game!{RESET}")
            self._reveal_roles(Camp.TOWNSFOLK)
            return

        print(
            f"\n{YELLOW}Townsfolk lead 3-{self.abyssal_score}!"
            f" But Rerir gets one chance to hunt Columbina...{RESET}"
        )
        rerir = next(p for p in self.players if p.role == Role.RERIR)
        townsfolk = [p for p in self.players if p.camp == Camp.TOWNSFOLK]

        target = rerir.hunt_columbina(townsfolk)
        print(f"\nRerir targets: {BOLD}{target.name}{RESET}")

        if target.role == Role.COLUMBINA:
            print(f"{BOLD}{RED}Rerir found Columbina! The Abyssals steal the victory!{RESET}")
            self._reveal_roles(Camp.ABYSSAL)
        else:
            print(f"{BOLD}{GREEN}Rerir guessed wrong! The Townsfolk win the game!{RESET}")
            self._reveal_roles(Camp.TOWNSFOLK)
