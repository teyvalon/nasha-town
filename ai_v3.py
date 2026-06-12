"""Joint-distribution Bayesian AI (v3): unified weighted-random decisions."""

from __future__ import annotations

import random
from itertools import combinations
from math import comb

from config import WAVE4_DOUBLE_FAIL_THRESHOLD, Camp, Role
from player import Player

# ── Constants ────────────────────────────────────────────────────────

SHARPEN = 2.5          # exponent for team-selection weight sharpening
VOTE_STRENGTH = 0.5    # how much a vote shifts beliefs (non-leader)
VOTE_STRENGTH_LEADER = VOTE_STRENGTH + 0.1  # same, for the team proposer



# ── Math ─────────────────────────────────────────────────────────────




def weighted_choice(options: list, weights: list[float]):
    """Pick from options with probability proportional to weights."""
    total = sum(weights)
    if total <= 0:
        return random.choice(options)
    r = random.random() * total
    cumul = 0.0
    for opt, w in zip(options, weights):
        cumul += w
        if r <= cumul:
            return opt
    return options[-1]


# ── Joint-distribution Bayesian AI ──────────────────────────────────


class JointBayesianAIPlayer(Player):
    """v3 AI: unified weighted-random decisions from joint P(evil_set).

    All roles share the same decision logic — only the joint prior differs.
    Team proposals and votes are weighted-random samples from utility,
    producing natural noise without special-case disguise mechanisms.
    """

    def __init__(self, name: str, role: Role):
        super().__init__(name, role)
        self._joint: dict[frozenset[Player], float] = {}
        self._players: list[Player] = []
        self._others: list[Player] = []
        self._allies: set[Player] = set()
        self._marginals: dict[Player, float] = {}
        self._marginals_dirty = True
        self._n = 0
        self._e = 0
        self._te = 0
        self._town_sc = 0
        self._abyss_sc = 0
        self._hunt: dict[Player, float] = {}  # running Columbina likelihood per townsfolk

    # ── Marginals ────────────────────────────────────────────────

    def _rebuild_marginals(self) -> None:
        if not self._marginals_dirty:
            return
        m: dict[Player, float] = {}
        for s, pr in self._joint.items():
            for p in s:
                m[p] = m.get(p, 0.0) + pr
        self._marginals = m
        self._marginals_dirty = False

    def _b(self, p: Player) -> float:
        if p is self:
            return 0.0  # cover belief: "I'm good"
        if not self._joint:
            return 0.5  # cover collapsed, no opinion
        if len(self._joint) == 1:
            return 0.999 if p in next(iter(self._joint)) else 0.001
        self._rebuild_marginals()
        return max(0.001, min(0.999, self._marginals.get(p, 0.0)))

    @property
    def _exposed(self) -> bool:
        """Cover belief collapsed — deterministic evidence proved us evil."""
        return self.camp == Camp.ABYSSAL and not self._joint

    # ── P(fail) ──────────────────────────────────────────────────

    def _fn(self, wave: int) -> int:
        return 2 if (wave == 3 and self._n >= WAVE4_DOUBLE_FAIL_THRESHOLD) else 1

    def _p_clean(self, team: list[Player]) -> float:
        """P(no evil on team) from joint distribution (cover: I'm good)."""
        total = 0.0
        for s, pr in self._joint.items():
            if not any(p in s for p in team):
                total += pr
        return total

    # ── Abyssal shared helpers ────────────────────────────────────

    def _urgency(self) -> float:
        """How aggressively Abyssals should act (0→passive, 1→all-in).

        Both scores contribute: town score drives desperation,
        abyss score drives momentum. Deeper games are always tenser.
        """
        return max(0.1, min(0.95, 0.1 + 0.35 * self._town_sc + 0.2 * self._abyss_sc))

    def _can_fail(self, team, wave) -> bool:
        """Can this team produce enough fakes to fail the mission?"""
        fn = self._fn(wave)
        evil_count = sum(1 for p in team if p is self or p in self._allies)
        return evil_count >= fn

    # ── Renormalize ──────────────────────────────────────────────

    def _renorm(self) -> None:
        if not self._joint:
            return
        total = sum(self._joint.values())
        if total < 1e-30:
            u = 1.0 / len(self._joint)
            self._joint = {k: u for k in self._joint}
        elif abs(total - 1) > 1e-10:
            inv = 1.0 / total
            self._joint = {k: v * inv for k, v in self._joint.items()}
        self._marginals_dirty = True

    # ── Init ─────────────────────────────────────────────────────

    def init_beliefs(self, players: list[Player], num_evil: int) -> None:
        self._players = list(players)
        self._others = [p for p in players if p is not self]
        self._n = len(players)
        self._e = num_evil
        # Everyone pretends to be good for beliefs: "there are E evil among others"
        self._te = num_evil
        self._allies = set(self.known_info.get("abyssal_allies", []))

        self._joint.clear()
        # Rerir: init hunt likelihoods for all non-allies
        if self.role == Role.RERIR:
            self._hunt = {p: 1.0 for p in self._players if p is not self and p not in self._allies}

        # Build joint prior — all roles start uniform (with Lauma filter)
        # For Abyssals this serves as their cover belief
        moon = set(self.known_info.get("moon_power_players", []))
        for combo in combinations(self._others, self._te):
            s = frozenset(combo)
            if self.role == Role.LAUMA and len(moon) == 2:
                if sum(1 for m in moon if m in s) != 1:
                    continue
            self._joint[s] = 1.0
        self._renorm()

    # ── Observe vote ─────────────────────────────────────────────

    def observe_team_vote(self, team, votes, wave, leader=None):
        # Continuous p_clean-weighted belief update for each voter.
        # Leader gets stronger adjustment (proposed this team = stronger signal).
        # YES vote: agreement = p_clean  (approving a clean team = good)
        # NO vote:  agreement = 1-p_clean (rejecting a dirty team = good)
        if len(self._joint) > 1:
            p_clean = self._p_clean(team)

            for v, yes in votes.items():
                if v is self:
                    continue
                agreement = p_clean if yes else (1 - p_clean)
                strength = VOTE_STRENGTH_LEADER if v is leader else VOTE_STRENGTH
                w = 1 - strength * (2 * agreement - 1)
                # Anomaly: on team but voted no, or off team but voted yes
                # Square the multiplier — "counts as two observations"
                on_team = v in team
                if (on_team and not yes) or (not on_team and yes):
                    w *= w
                for s in self._joint:
                    if v in s:
                        self._joint[s] *= w
            self._renorm()

        # Rerir: update hunt likelihoods from vote + proposal accuracy
        # Swing grows with wave: early votes are noisy, late votes are telling
        if self.role == Role.RERIR and self._hunt:
            boost = 1.3 + 0.2 * wave   # wave 0→1.3, 1→1.5, 2→1.7, 3→1.9
            penalty = 1.0 / boost       # inverse
            evil = self._allies | {self}
            has_evil = any(p in evil for p in team)
            for v, yes in votes.items():
                if v not in self._hunt:
                    continue
                correct = (has_evil and not yes) or (not has_evil and yes)
                # Anomaly: on team but voted no, or off team but voted yes
                # If this anomaly is also "correct", it's a strong Columbina signal
                on_team = v in team
                anomaly = (on_team and not yes) or (not on_team and yes)
                if correct:
                    self._hunt[v] *= (boost * boost) if anomaly else boost
                else:
                    self._hunt[v] *= penalty
            if leader in self._hunt:
                clean = not has_evil
                self._hunt[leader] *= boost if clean else penalty

    # ── Observe mission ──────────────────────────────────────────

    def observe_mission_result(self, team, num_fakes, wave, leader=None):
        fn = self._fn(wave)
        failed = num_fakes >= fn

        if failed:
            self._abyss_sc += 1
        else:
            self._town_sc += 1

        if len(self._joint) > 1:
            team_set = frozenset(team)
            # Base fake rate: early game evil hides more, late game must act
            q_base = 0.5 + 0.1 * min(wave, 3)  # wave 0→0.5, 1→0.6, 2→0.7, 3→0.8
            q_leader = min(q_base + 0.2, 0.95)  # evil leader picked this team to act
            f = num_fakes

            # Probabilistic: binomial model
            for s in list(self._joint):
                evil_on_team = s & team_set
                k = len(evil_on_team)
                if k < f or (k == 0 and f > 0):
                    del self._joint[s]
                else:
                    is_leader_evil = leader is not None and leader in evil_on_team
                    if is_leader_evil:
                        q_eff = (q_leader + q_base * (k - 1)) / k
                    else:
                        q_eff = q_base
                    self._joint[s] *= comb(k, f) * (q_eff ** f) * ((1 - q_eff) ** (k - f))

            if self.role == Role.COLUMBINA:
                self._columbina_reveal(wave)

            self._renorm()

    # ── Columbina gradual reveal ─────────────────────────────────

    def _columbina_reveal(self, wave: int) -> None:
        """Columbina gradually trusts her private knowledge of who is evil.

        Each wave, penalise hypotheses that include known-good players.
        Strength rises with wave so early game she blends in, late game she leads.
        """
        known_evil = set(self.known_info.get("known_abyssals", []))
        if not known_evil:
            return
        # strength: wave 0→0.05, 1→0.27, 2→0.49, 3→0.71, 4→0.93
        strength = 0.05 + 0.22 * wave
        penalty = 1 - strength
        for s in self._joint:
            wrong = sum(1 for p in s if p not in known_evil)
            if wrong > 0:
                self._joint[s] *= penalty ** wrong

    # ── Propose team (unified weighted random) ───────────────────

    def propose_team(self, players, team_size, wave):
        others = [p for p in players if p is not self]
        teams = []
        weights = []

        if self._exposed:
            # Cover blown — prefer teams that can fail, uniform otherwise
            for combo in combinations(others, team_size - 1):
                team = [self] + list(combo)
                w = 1.0 if self._can_fail(team, wave) else 0.001
                teams.append(team)
                weights.append(w)
        else:
            # Abyssal leader: penalise teams that can't fail the mission.
            urg = self._urgency() if self.camp == Camp.ABYSSAL else 0.0

            for combo in combinations(others, team_size - 1):
                team = [self] + list(combo)
                p_clean = self._p_clean(team)
                w = max(0.001, p_clean ** SHARPEN)
                if urg > 0 and not self._can_fail(team, wave):
                    w *= (1 - urg)
                    w = max(0.001, w)
                teams.append(team)
                weights.append(w)

        best = weighted_choice(teams, weights) if teams else [self] + random.sample(others, team_size - 1)
        random.shuffle(best)
        return best

    # ── Vote (unified) ───────────────────────────────────────────

    def vote(self, team, wave, proposal_num, max_proposals, leader=None):
        if proposal_num >= max_proposals - 1:
            return True

        # Lauma structural rule: reject if both moon-power on team
        if self.role == Role.LAUMA:
            moon = set(self.known_info.get("moon_power_players", []))
            if sum(1 for p in moon if p in team) >= 2:
                return False

        # Exposed Abyssal: no cover to maintain
        if self._exposed:
            if self in team:
                return self._can_fail(team, wave)
            # Not on team: urgency decides whether to help allies or obstruct
            urg = self._urgency()
            if random.random() < urg:
                return self._can_fail(team, wave)  # help allies when desperate
            # Ally's proposal → support; enemy's → mostly reject
            if leader in self._allies:
                return True
            return random.random() < urg * 0.3

        p_clean = self._p_clean(team)

        # Abyssal strategic voting
        # Urgency drives how often to override cover vote with strategic vote.
        # Strategic vote approves iff the team can actually fail the mission
        # (on double-fail waves this requires 2+ evil, not just 1).
        if self.camp == Camp.ABYSSAL:
            urg = self._urgency()
            if random.random() < urg:
                return self._can_fail(team, wave)

        # Leader boost: mild compensation for propose sharpening
        if leader is self:
            p_vote = p_clean ** (1 / SHARPEN)
        else:
            p_vote = p_clean
        return random.random() < p_vote

    # ── Place bomb ───────────────────────────────────────────────

    def place_bomb(self, wave, team=None):
        if self.camp == Camp.TOWNSFOLK:
            return True

        # Exposed: no cover to protect, always fake if possible
        if self._exposed:
            return False if self._can_fail(team or [], wave) else True

        # Can't produce enough fakes → place real
        if not self._can_fail(team or [], wave):
            return True

        urg = self._urgency()
        allies_here = sum(1 for p in (team or []) if p in self._allies)

        # Urgent: town about to win → always fake
        if self._town_sc >= 2:
            if (self.role == Role.RERIR and self._abyss_sc < 2
                    and allies_here > 0 and self._columbina_confidence() >= 0.9):
                return True
            return False

        # Normal: probability-based with urgency lerp
        base = {Role.RERIR: 0.85, Role.DOTTORE: 0.50}.get(self.role, 0.70)  # [0.50, 0.85]
        evil_here = allies_here + 1  # including self; [1, num_evil]
        exposure = evil_here / (len(team) if team else 3)  # (0, 1]; typically 0.20–0.67
        base_fake = base * (1 - exposure)  # [0, 0.85); typically 0.23–0.57
        fake_prob = base_fake + urg * (0.95 - base_fake)  # [base_fake, 0.95]
        return random.random() >= fake_prob

    # ── Hunt Columbina ───────────────────────────────────────────

    def _columbina_scores(self) -> dict[Player, float]:
        """Normalised hunt likelihoods — higher = more likely Columbina."""
        if not self._hunt:
            return {}
        total = sum(self._hunt.values()) or 1.0
        return {p: v / total for p, v in self._hunt.items()}

    def _columbina_confidence(self) -> float:
        scores = self._columbina_scores()
        return max(scores.values()) if scores else 0.0

    def hunt_columbina(self, townsfolk):
        scores = self._columbina_scores()
        if not scores:
            return random.choice(townsfolk)
        weights = [max(0.001, scores.get(p, 0.001)) ** SHARPEN for p in townsfolk]
        return weighted_choice(townsfolk, weights)
