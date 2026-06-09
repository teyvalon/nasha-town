"""Bayesian AI player (v2): probability-based reasoning and decision making."""

from __future__ import annotations

import math
import random
from itertools import combinations

from config import WAVE4_DOUBLE_FAIL_THRESHOLD, Camp, Role
from player import Player

# ── Fake-bomb probability model ──────────────────────────────────────
# Per-Abyssal probability of placing a fake bomb (as perceived by observers).
# This is a simplified estimate used for belief updates — the actual AI
# decision is more nuanced (see place_bomb), but observers don't know that.
ASSUMED_FAKE_RATE = 0.65  # average Abyssal fakes ~65% of the time (observer estimate)

# ── Hyperparameters ──────────────────────────────────────────────────

ALPHA = 5.0  # good-player vote sensitivity
BETA = 5.0  # evil-player vote sensitivity
EPSILON = 0.05  # voting noise floor
LAMBDA = 0.3  # weight of P_accept in team utility
TAU_INIT = 0.5  # initial success-probability threshold for approving
TAU_MIN = 0.3  # minimum threshold after consecutive failures

# ── Columbina disguise parameters ────────────────────────────────
# She knows everything but must act like she doesn't.
# disguise_rate decays from DISGUISE_EARLY to DISGUISE_LATE over 5 waves.
DISGUISE_EARLY = 0.40  # early game: 40% chance to act "dumb"
DISGUISE_LATE = 0.12   # late game: 12% (must actually help now)


# ── Math helpers ─────────────────────────────────────────────────────


def _comb(n: int, r: int) -> int:
    """Binomial coefficient C(n, r)."""
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def _p_fakes_given_k(f: int, k: int, q: float = ASSUMED_FAKE_RATE) -> float:
    """P(exactly f fake bombs | k Abyssals on team).

    Each of k Abyssals independently fakes with probability q.
    Good players always place real bombs, so f can't exceed k.
    """
    if f > k or f < 0:
        return 0.0
    return _comb(k, f) * (q ** f) * ((1.0 - q) ** (k - f))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _clamp(b: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, b))


def _prob_to_logodds(b: float) -> float:
    b = _clamp(b)
    return math.log(b / (1.0 - b))


def _logodds_to_prob(ell: float) -> float:
    return _clamp(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, ell)))))


# ── Bayesian AI ──────────────────────────────────────────────────────


class BayesianAIPlayer(Player):
    """Advanced AI using Bayesian belief updates (v2).

    Maintains a per-player evil-probability via log-odds, updated after
    every vote and mission.  Decisions maximise expected utility.
    """

    def __init__(self, name: str, role: Role):
        super().__init__(name, role)
        self._logodds: dict[Player, float] = {}
        self._all_players: list[Player] = []
        self._num_players = 0
        self._num_evil = 0
        self._tau = TAU_INIT
        self._consecutive_fails = 0
        self._town_score = 0
        self._abyssal_score = 0
        self._vote_history: list[tuple[list[Player], dict[Player, bool]]] = []

    # ── Belief helpers ───────────────────────────────────────────

    def _b(self, p: Player) -> float:
        """Current evil probability for *p*."""
        if p is self:
            return 1.0 if self.camp == Camp.ABYSSAL else 0.0
        return _logodds_to_prob(self._logodds.get(p, 0.0))

    def _normalize_beliefs(self) -> None:
        """Adjust beliefs so they respect the global constraint: exactly E evil.

        Without this, identifying 2 evils in a 5p/2e game leaves the other 3
        players stuck at their prior (~0.5) instead of being cleared (~0).
        """
        others = [p for p in self._all_players if p is not self]
        if not others:
            return

        # How many evil should be among 'others'?
        target = self._num_evil
        if self.camp == Camp.ABYSSAL:
            target -= 1  # exclude self

        # Split into locked (near-certain) and uncertain
        LOCK_HI = 0.99
        LOCK_LO = 0.01
        locked_evil = 0
        uncertain: list[Player] = []

        for p in others:
            b = self._b(p)
            if b >= LOCK_HI:
                locked_evil += 1
            elif b <= LOCK_LO:
                pass  # locked good
            else:
                uncertain.append(p)

        remaining_evil = max(0.0, target - locked_evil)

        if not uncertain:
            return

        # Scale uncertain beliefs so they sum to remaining_evil
        current_sum = sum(self._b(p) for p in uncertain)
        if current_sum < 1e-10:
            # All uncertain are near zero; distribute evenly
            if remaining_evil > 0:
                even_b = remaining_evil / len(uncertain)
                for p in uncertain:
                    self._logodds[p] = _prob_to_logodds(even_b)
            return

        raw_scale = remaining_evil / current_sum
        # Dampen: don't snap fully, blend toward the ideal scale
        # Fewer players → stronger normalization effect → more damping needed
        damp = 0.5
        scale = 1.0 + damp * (raw_scale - 1.0)
        for p in uncertain:
            old_b = self._b(p)
            new_b = _clamp(old_b * scale)
            self._logodds[p] = _prob_to_logodds(new_b)

    def _fails_needed(self, wave: int) -> int:
        if wave == 3 and self._num_players >= WAVE4_DOUBLE_FAIL_THRESHOLD:
            return 2
        return 1

    # ── Probability computations ─────────────────────────────────

    def _p_fakes(self, team: list[Player], wave: int, exact_f: int) -> float:
        """P(exactly exact_f fake bombs) for a given team."""
        n = len(team)
        total = 0.0
        for mask in range(1 << n):
            prob = 1.0
            k = 0
            for i in range(n):
                bi = self._b(team[i])
                if mask & (1 << i):
                    prob *= bi
                    k += 1
                else:
                    prob *= 1.0 - bi
            total += prob * _p_fakes_given_k(exact_f, k)
        return total

    def _p_fail(self, team: list[Player], wave: int) -> float:
        """P(mission fails) for a given team, based on current beliefs."""
        n = len(team)
        fn = self._fails_needed(wave)
        # Sum over all fake counts that cause failure
        return sum(self._p_fakes(team, wave, f) for f in range(fn, n + 1))

    def _p_accept(self, team: list[Player], wave: int) -> float:
        """P(proposal gets majority approval), given current beliefs."""
        p_fail = self._p_fail(team, wave)
        p_succ = 1.0 - p_fail

        p_good_yes = EPSILON + (1.0 - 2 * EPSILON) * _sigmoid(ALPHA * (p_succ - 0.5))
        p_evil_yes = EPSILON + (1.0 - 2 * EPSILON) * _sigmoid(BETA * (p_fail - 0.5))

        # Per-player approval probability
        p_yes: list[float] = []
        for p in self._all_players:
            bi = self._b(p)
            py = (1.0 - bi) * p_good_yes + bi * p_evil_yes
            if p in team:
                py = max(py, 0.7)  # team members lean toward approval
            p_yes.append(py)

        # DP to compute P(>= threshold votes)
        n = len(p_yes)
        threshold = n // 2 + 1
        dp = [0.0] * (n + 1)
        dp[0] = 1.0
        for py in p_yes:
            new_dp = [0.0] * (n + 1)
            for j in range(n + 1):
                if dp[j] == 0.0:
                    continue
                if j + 1 <= n:
                    new_dp[j + 1] += dp[j] * py
                new_dp[j] += dp[j] * (1.0 - py)
            dp = new_dp
        return sum(dp[threshold:])

    # ── Lifecycle hooks ──────────────────────────────────────────

    def init_beliefs(self, players: list[Player], num_evil: int) -> None:
        self._all_players = list(players)
        self._num_players = len(players)
        self._num_evil = num_evil

        N = self._num_players
        E = self._num_evil

        for j in players:
            if j is self:
                continue

            if self.role == Role.COLUMBINA:
                abyssals = self.known_info.get("known_abyssals", [])
                b = 0.999 if j in abyssals else 0.001

            elif self.role == Role.LAUMA:
                moon = self.known_info.get("moon_power_players", [])
                if j in moon:
                    b = 0.5
                else:
                    b = _clamp((E - 1) / max(1, N - 3))

            elif self.camp == Camp.ABYSSAL:
                allies = self.known_info.get("abyssal_allies", [])
                b = 0.999 if j in allies else 0.001

            else:  # generic townsfolk
                b = _clamp(E / (N - 1))

            self._logodds[j] = _prob_to_logodds(b)

    # ── Observation updates ──────────────────────────────────────

    def observe_team_vote(self, team, votes, wave):
        p_fail = self._p_fail(team, wave)
        p_succ = 1.0 - p_fail

        p_good_yes = EPSILON + (1.0 - 2 * EPSILON) * _sigmoid(ALPHA * (p_succ - 0.5))
        p_evil_yes = EPSILON + (1.0 - 2 * EPSILON) * _sigmoid(BETA * (p_fail - 0.5))

        for voter, voted_yes in votes.items():
            if voter is self:
                continue
            if voted_yes:
                lr = p_evil_yes / max(p_good_yes, 1e-10)
            else:
                lr = (1.0 - p_evil_yes) / max(1.0 - p_good_yes, 1e-10)
            if lr > 0:
                self._logodds[voter] = self._logodds.get(voter, 0.0) + math.log(lr)

        self._vote_history.append((list(team), dict(votes)))
        self._normalize_beliefs()

    def observe_mission_result(self, team, num_fakes, wave):
        fails_needed = self._fails_needed(wave)
        failed = num_fakes >= fails_needed

        if failed:
            self._consecutive_fails += 1
            self._abyssal_score += 1
        else:
            self._consecutive_fails = 0
            self._town_score += 1
        self._tau = max(TAU_MIN, TAU_INIT - 0.05 * self._consecutive_fails)

        # Update beliefs using exact fake count, not just pass/fail
        for j in team:
            if j is self:
                continue

            others = [p for p in team if p is not j and p is not self]
            n_others = len(others)

            # Count self as known evil if applicable
            self_evil_in_team = self in team and self.camp == Camp.ABYSSAL
            self_k = 1 if self_evil_in_team else 0

            # P(exactly num_fakes | j is evil) and P(exactly num_fakes | j is good)
            p_obs_if_evil = 0.0
            p_obs_if_good = 0.0

            for mask in range(1 << n_others):
                prob = 1.0
                k = 0
                for i in range(n_others):
                    bi = self._b(others[i])
                    if mask & (1 << i):
                        prob *= bi
                        k += 1
                    else:
                        prob *= 1.0 - bi

                # If j is evil: k_total = k + self_k + 1
                p_obs_if_evil += prob * _p_fakes_given_k(num_fakes, k + self_k + 1)
                # If j is good: k_total = k + self_k
                p_obs_if_good += prob * _p_fakes_given_k(num_fakes, k + self_k)

            num = max(p_obs_if_evil, 1e-10)
            den = max(p_obs_if_good, 1e-10)
            lr = num / den
            if lr > 0:
                self._logodds[j] = self._logodds.get(j, 0.0) + math.log(lr)

        self._normalize_beliefs()

    # ── Columbina disguise ───────────────────────────────────────

    def _disguise_rate(self, wave: int) -> float:
        """How likely Columbina is to make a deliberate 'mistake' this wave."""
        # Linear decay from DISGUISE_EARLY (wave 0) to DISGUISE_LATE (wave 4)
        t = wave / 4.0
        return DISGUISE_EARLY * (1.0 - t) + DISGUISE_LATE * t

    # ── Propose team ─────────────────────────────────────────────

    def propose_team(self, players, team_size, wave):
        others = [p for p in players if p is not self]
        best_team: list[Player] | None = None
        best_u = -1.0

        for combo in combinations(others, team_size - 1):
            team = [self] + list(combo)

            if self.camp == Camp.TOWNSFOLK:
                p_succ = 1.0 - self._p_fail(team, wave)
                p_acc = self._p_accept(team, wave)
                u = p_succ * (p_acc ** LAMBDA)
            else:
                # Evil: need at least one confirmed Abyssal on team
                allies = set(self.known_info.get("abyssal_allies", []))
                if not (self in team or any(p in allies for p in team)):
                    continue
                p_fail = self._p_fail(team, wave)
                p_acc = self._p_accept(team, wave)
                u = p_fail * (p_acc ** LAMBDA)

            if u > best_u:
                best_u = u
                best_team = team

        if best_team is None:
            best_team = [self] + random.sample(others, team_size - 1)

        # Columbina disguise: occasionally propose a "normal-looking" team
        # that includes one Abyssal, as a regular townsfolk might do
        if self.role == Role.COLUMBINA and random.random() < self._disguise_rate(wave):
            abyssals = self.known_info.get("known_abyssals", [])
            townsfolk = [p for p in others if p not in abyssals]
            if abyssals and len(townsfolk) >= team_size - 2:
                decoy = random.choice(abyssals)
                fill = random.sample(townsfolk, team_size - 2)
                best_team = [self, decoy] + fill

        random.shuffle(best_team)
        return best_team

    # ── Vote ─────────────────────────────────────────────────────

    def vote(self, team, wave, proposal_num, max_proposals):
        if proposal_num >= max_proposals - 1:
            return True

        if self.role == Role.COLUMBINA:
            abyssals = set(self.known_info.get("known_abyssals", []))
            has_evil = any(p in abyssals for p in team)
            true_vote = not has_evil
            # Only disguise when the stakes aren't critical
            # Never approve a bad team when Abyssals are close to winning
            can_disguise = self._abyssal_score < 2
            if can_disguise and random.random() < self._disguise_rate(wave):
                return not true_vote
            return true_vote

        if self.camp == Camp.TOWNSFOLK:
            p_succ = 1.0 - self._p_fail(team, wave)
            return p_succ >= self._tau
        else:
            # Abyssal voting: blend "want mission to fail" with
            # "act like a good player to avoid detection"
            p_fail = self._p_fail(team, wave)
            evil_desire = _sigmoid(BETA * (p_fail - 0.5))
            # Also compute what a good player would vote
            p_succ = 1.0 - p_fail
            good_cover = 1.0 if p_succ >= self._tau else 0.0
            # Blend: smaller games need more cover
            cover_weight = 0.3 if self._num_players <= 5 else 0.15
            prob_yes = (1.0 - cover_weight) * evil_desire + cover_weight * good_cover
            if random.random() < 0.1:
                return random.choice([True, False])
            return random.random() < prob_yes

    # ── Place bomb ───────────────────────────────────────────────

    def place_bomb(self, wave, team=None):
        if self.camp == Camp.TOWNSFOLK:
            return True

        need_double = wave == 3 and self._num_players >= WAVE4_DOUBLE_FAIL_THRESHOLD

        if need_double:
            # Wave 4 special (7+ players): need 2+ fakes to score
            allies = set(self.known_info.get("abyssal_allies", []))
            allies_on_team = sum(1 for p in (team or []) if p in allies)
            if allies_on_team >= 1:
                return False  # ally + self = 2 fakes, go for it
            else:
                # Solo: can't reach 2 fakes alone, play real to stay hidden
                return True

        # Urgency: town at 2, one more success wins
        if self._town_score >= 2:
            # Rerir gambit: if an ally is also on the team (they'll fake),
            # Rerir can play real and bet on the Moon Hunt.
            # Only when Abyssals aren't also at match point.
            if self.role == Role.RERIR and self._abyssal_score < 2 and team:
                allies = set(self.known_info.get("abyssal_allies", []))
                ally_on_team = any(p in allies for p in team)
                if ally_on_team:
                    conf = self._columbina_confidence()
                    if conf >= 0.6:
                        return True  # real bomb — ally covers, go for hunt
            return False

        # ── Smart bomb decision ────────────────────────────────────
        #
        # fake_prob = base × (1 - exposure) × urgency
        #
        # base:     role personality — how eager to sabotage
        # exposure: how much blame falls on you if mission fails
        # urgency:  how badly the team needs this point
        #
        allies = set(self.known_info.get("abyssal_allies", []))
        team_size = len(team) if team else 3
        allies_on_team = sum(1 for p in (team or []) if p in allies)

        # 1) Base desire by role
        if self.role == Role.RERIR:
            base = 0.85
        elif self.role == Role.DOTTORE:
            base = 0.50  # must maintain "False Moon" cover
        else:
            base = 0.70

        # 2) Exposure risk: smaller team = more blame per person
        #    ally present = blame is shared, less risky to play real
        #    fewer total players = getting caught is more devastating
        blame_share = 1.0 / team_size  # 2-man: 0.5, 5-man: 0.2
        game_caution = 1.0 + max(0, 6 - self._num_players) * 0.15  # 5p: 1.15, 6p: 1.0
        if allies_on_team > 0:
            exposure = blame_share * 0.3 * game_caution
        else:
            exposure = blame_share * game_caution

        # 3) Urgency: how much do we need to score?
        score_gap = self._town_score - self._abyssal_score
        if score_gap >= 1:
            urgency = 1.2  # falling behind, push harder
        elif score_gap <= -1:
            urgency = 0.8  # ahead, can afford to lay low
        else:
            urgency = 1.0  # tied

        fake_prob = min(0.95, base * (1.0 - exposure) * urgency)

        return random.random() >= fake_prob  # True = real, False = fake

    # ── Hunt Columbina ───────────────────────────────────────────

    def _columbina_scores(self) -> dict[Player, float]:
        """Score each townsfolk on how Columbina-like their voting is.

        Higher = more likely to be Columbina (votes correlate with
        knowing who the Abyssals are).
        """
        allies = set(self.known_info.get("abyssal_allies", []))
        evil_set = allies | {self}
        townsfolk = [p for p in self._all_players if p not in evil_set]

        scores: dict[Player, float] = {}
        for tf in townsfolk:
            correct = 0
            total = 0
            for team, votes in self._vote_history:
                if tf not in votes:
                    continue
                total += 1
                has_evil = any(p in evil_set for p in team)
                voted_yes = votes[tf]
                if has_evil and not voted_yes:
                    correct += 1
                elif not has_evil and voted_yes:
                    correct += 1
            scores[tf] = correct / max(total, 1)
        return scores

    def _columbina_confidence(self) -> float:
        """How confident Rerir is about identifying Columbina.

        Returns a value in [0, 1]. High when the top candidate is clearly
        separated from the rest AND enough votes have been observed.
        """
        scores = self._columbina_scores()
        if len(scores) < 2 or not self._vote_history:
            return 0.0

        ranked = sorted(scores.values(), reverse=True)
        gap = ranked[0] - ranked[1]  # separation between #1 and #2
        # Scale by how many vote rounds we've seen (more data = more trust)
        data_factor = min(1.0, len(self._vote_history) / 6.0)
        return gap * data_factor

    def hunt_columbina(self, townsfolk):
        """Rerir picks the townsfolk whose votes most resemble 'knows evil'."""
        scores = self._columbina_scores()
        if not scores:
            return random.choice(townsfolk)
        return max(townsfolk, key=lambda p: scores.get(p, 0.0))
