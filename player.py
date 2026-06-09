"""Player classes: human input and AI placeholder logic."""

from __future__ import annotations

import random

from config import Camp, Role, WAVE4_DOUBLE_FAIL_THRESHOLD


class Player:
    """Base class for all players."""

    def __init__(self, name: str, role: Role):
        self.name = name
        self.role = role
        self.camp = role.camp
        # Information revealed during night phase, populated by Game.
        # Possible keys: "abyssal_allies", "known_abyssals", "moon_power_players"
        self.known_info: dict[str, list[Player]] = {}

    def propose_team(
        self, players: list[Player], team_size: int, wave: int
    ) -> list[Player]:
        """Leader proposes a team for the current wave."""
        raise NotImplementedError

    def vote(
        self, team: list[Player], wave: int, proposal_num: int, max_proposals: int
    ) -> bool:
        """Vote approve (True) or reject (False) on a proposed team."""
        raise NotImplementedError

    def place_bomb(self, wave: int) -> bool:
        """Place a bomb: True = real, False = fake."""
        raise NotImplementedError

    def hunt_columbina(self, townsfolk: list[Player]) -> Player:
        """Rerir's end-game ability: pick who is Columbina among townsfolk."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class HumanPlayer(Player):
    """Interactive human player via CLI."""

    def propose_team(self, players, team_size, wave):
        print(f"\nYou are the leader! Propose a team of {team_size} for Wave {wave + 1}.")
        print("Available players:")
        for i, p in enumerate(players):
            tag = " (you)" if p is self else ""
            print(f"  {i + 1}. {p.name}{tag}")

        while True:
            try:
                raw = input(f"Enter {team_size} player numbers separated by spaces: ").strip()
                indices = [int(x) - 1 for x in raw.split()]
                if len(indices) != team_size:
                    print(f"Please select exactly {team_size} players.")
                    continue
                if any(i < 0 or i >= len(players) for i in indices):
                    print("Invalid player number.")
                    continue
                if len(set(indices)) != len(indices):
                    print("No duplicate players.")
                    continue
                return [players[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid input, try again.")

    def vote(self, team, wave, proposal_num, max_proposals):
        while True:
            choice = input("Approve this team? (y/n): ").strip().lower()
            if choice in ("y", "yes"):
                return True
            if choice in ("n", "no"):
                return False
            print("Enter 'y' or 'n'.")

    def place_bomb(self, wave):
        if self.camp == Camp.TOWNSFOLK:
            print("You place a real bomb. (Townsfolk must place real bombs)")
            return True
        # Abyssal human gets a choice
        while True:
            choice = input("Place a [r]eal or [f]ake bomb? ").strip().lower()
            if choice in ("r", "real"):
                return True
            if choice in ("f", "fake"):
                return False
            print("Enter 'r' or 'f'.")

    def hunt_columbina(self, townsfolk):
        print("\nYou are Rerir! Identify who you think is Columbina:")
        for i, p in enumerate(townsfolk):
            print(f"  {i + 1}. {p.name}")
        while True:
            try:
                idx = int(input("Enter player number: ").strip()) - 1
                if 0 <= idx < len(townsfolk):
                    return townsfolk[idx]
                print("Invalid number.")
            except ValueError:
                print("Invalid input.")


class AIPlayer(Player):
    """Role-aware AI (v1).

    - Propose: always include self; role-specific filtering
    - Vote: role-specific approval logic
    - Bomb: townsfolk real, abyssal fake
    - Hunt: random guess (placeholder)
    """

    # ── Propose ───────────────────────────────────────────────────

    def propose_team(self, players, team_size, wave):
        others = [p for p in players if p is not self]
        must_include: list[Player] = []
        excluded: set[Player] = set()

        if self.role == Role.COLUMBINA:
            # Never include any known abyssal
            excluded = set(self.known_info.get("known_abyssals", []))

        elif self.camp == Camp.ABYSSAL:
            allies = self.known_info.get("abyssal_allies", [])
            need_double = (
                wave == 3 and len(players) >= WAVE4_DOUBLE_FAIL_THRESHOLD
            )
            if need_double and allies:
                # Wave 4 double-fail: bring an ally
                must_include = [random.choice(allies)]
            else:
                # Otherwise keep allies out
                excluded = set(allies)

        elif self.role == Role.LAUMA:
            # Don't include both moon-power players at the same time
            moon = self.known_info.get("moon_power_players", [])
            if len(moon) >= 2:
                excluded = {random.choice(moon)}

        # Build team: self + must_include + random fill from pool
        team: list[Player] = [self] + must_include
        pool = [p for p in others if p not in excluded and p not in team]
        remaining = team_size - len(team)

        if remaining > 0:
            if len(pool) >= remaining:
                team += random.sample(pool, remaining)
            else:
                team += pool
                fallback = [p for p in others if p not in team]
                team += random.sample(fallback, remaining - len(pool))

        random.shuffle(team)
        return team

    # ── Vote ──────────────────────────────────────────────────────

    def vote(self, team, wave, proposal_num, max_proposals):
        # Last proposal — must approve to avoid Abyssal auto-win
        if proposal_num >= max_proposals - 1:
            return True

        if self.role == Role.COLUMBINA:
            # Only approve all-good teams
            abyssals = set(self.known_info.get("known_abyssals", []))
            return not any(p in abyssals for p in team)

        if self.role == Role.LAUMA:
            if self not in team:
                return False
            # Reject if both moon-power players are in team
            moon = set(self.known_info.get("moon_power_players", []))
            if sum(1 for p in team if p in moon) >= 2:
                return False
            return True

        if self.camp == Camp.TOWNSFOLK:
            # Generic townsfolk: approve iff self in team
            return self in team

        # Abyssal: approve if any abyssal (self or ally) is on the team
        allies = set(self.known_info.get("abyssal_allies", []))
        return self in team or any(p in allies for p in team)

    # ── Bomb ──────────────────────────────────────────────────────

    def place_bomb(self, wave):
        if self.camp == Camp.TOWNSFOLK:
            return True
        return False  # Abyssal always fakes

    # ── Hunt ──────────────────────────────────────────────────────

    def hunt_columbina(self, townsfolk):
        return random.choice(townsfolk)
