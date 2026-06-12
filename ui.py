"""Right-side belief panel using ANSI escape codes.

Drawn only when the game pauses for user input, so no scrolling conflicts.
"""

from __future__ import annotations

import os
import sys

from player import Player

_CSI = "\033["

DIM = f"{_CSI}2m"
BOLD = f"{_CSI}1m"
RESET = f"{_CSI}0m"
RED = f"{_CSI}91m"
GREEN = f"{_CSI}92m"
YELLOW = f"{_CSI}93m"


def _move(row: int, col: int) -> str:
    return f"{_CSI}{row};{col}H"


def _clear_to_eol() -> str:
    return f"{_CSI}K"


def _save_cursor() -> str:
    return f"{_CSI}s"


def _restore_cursor() -> str:
    return f"{_CSI}u"


class BeliefPanel:
    """Right-side belief panel, redrawn before each user input.

    Caches the latest observer data via update(). When show() is called
    (right before input()), it paints the panel anchored to the bottom-right,
    then restores the cursor so input() appears in the correct place.
    """

    PANEL_WIDTH = 33

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._rows = 24
        self._cols = 80
        self._panel_col = 52
        # (name, subtitle, players, b_func)
        self._observers: list[tuple[str, str, list[Player], callable]] = []
        self._painted = False

    def setup(self) -> None:
        if not self.enabled:
            return
        self._update_size()

    def cleanup(self) -> None:
        if not self.enabled or not self._painted:
            return
        self._update_size()
        self._clear_panel()
        self._painted = False

    def _update_size(self) -> None:
        try:
            size = os.get_terminal_size()
            self._rows = size.lines
            self._cols = size.columns
        except OSError:
            pass
        self._panel_col = max(1, self._cols - self.PANEL_WIDTH + 1)

    def _clear_panel(self) -> None:
        col = self._panel_col
        buf = [_save_cursor()]
        for row in range(1, self._rows + 1):
            buf.append(_move(row, col) + _clear_to_eol())
        buf.append(_restore_cursor())
        sys.stdout.write("".join(buf))
        sys.stdout.flush()

    def update(
        self,
        observers: list[tuple[str, str, list[Player], callable]],
    ) -> None:
        """Cache latest beliefs. Actual drawing happens in show().

        observers: list of (name, subtitle, players, b_func) tuples.
        """
        if not self.enabled:
            return
        self._observers = observers

    def show(self) -> None:
        """Paint the panel. Call this right before input()."""
        if not self.enabled or not self._observers:
            return
        self._update_size()

        col = self._panel_col
        w = self.PANEL_WIDTH

        lines: list[str] = []
        for entry in self._observers:
            obs_name, subtitle, players, b_func = entry
            lines.append(f"{'─' * w}")
            label = f" {obs_name}"
            if subtitle:
                label += f" {DIM}{subtitle}{RESET}"
            lines.append(f"{BOLD}{label}{RESET}")
            for p in players:
                if p.name == obs_name:
                    continue
                b = b_func(p)
                name = p.name[:15].ljust(15)
                bar_len = int(b * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                if b >= 0.7:
                    color = RED
                elif b >= 0.4:
                    color = YELLOW
                else:
                    color = GREEN
                lines.append(f"  {name}{color}{bar}{RESET}{b:>4.0%}")
        lines.append(f"{'─' * w}")

        max_row = self._rows - 1
        start_row = max(1, max_row - len(lines) + 1)

        buf = [_save_cursor()]
        # Clear panel region
        for row in range(1, max_row + 1):
            buf.append(_move(row, col) + _clear_to_eol())
        # Paint
        for i, line in enumerate(lines):
            row = start_row + i
            if row > max_row:
                break
            buf.append(_move(row, col) + line)
        buf.append(_restore_cursor())

        sys.stdout.write("".join(buf))
        sys.stdout.flush()
        self._painted = True

    def hide(self) -> None:
        """Clear the panel after input() returns, before more print()s."""
        if not self.enabled or not self._painted:
            return
        self._update_size()
        self._clear_panel()
        self._painted = False
