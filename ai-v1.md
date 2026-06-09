# AI Algorithm v1 — Role-Aware Basic Logic

First version of the AI player strategy for TEYVALON.

## Propose Team

1. Always include self in the team.
2. **Abyssal**: exclude other Abyssals from the team, UNLESS it's Wave 4 with 7+ players (double-fail rule) — in that case, actively include one ally.
3. **Columbina (Moon Goddess)**: exclude all known Abyssals from the team.
4. **Lauma (Moonchanter)**: exclude one of the two moon-power players randomly, to ensure both Columbina and Dottore are never on the same team.
5. **Generic Townsfolk**: no special filtering, random fill.

## Vote

0. **All roles**: on the last proposal of a wave, always approve (to avoid Abyssal auto-win from all proposals being rejected).
1. **Generic Townsfolk**: approve if self is on the team; otherwise reject.
2. **Lauma**: same as generic Townsfolk, plus reject any team that includes both moon-power players.
3. **Columbina**: only approve teams where all members are good (no known Abyssals).
4. **Abyssal**: approve if at least one Abyssal (self or ally) is on the team.

## Place Bomb

1. **Townsfolk**: always place real bomb.
2. **Abyssal**: always place fake bomb.

## Hunt Columbina (Rerir)

Random guess among Townsfolk (placeholder — to be improved).
