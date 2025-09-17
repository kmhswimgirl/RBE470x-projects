# This is necessary to find the main code
import sys
import math
import random

sys.path.insert(0, '../Bomberman')
# Import necessary stuff
from entity import CharacterEntity # type: ignore
from colorama import Fore, Back

from collections import deque  # minimal extra import


# Flood-to-goal + MDP (value iteration) w safety

INF = 10**9
DIRS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
DIRS9 = [(0, 0)] + DIRS8  # allow "stay" action

# MDP params
DISCOUNT = 0.95
P_INTEND = 0.80           # chance intended move happens; rest spreads to other legal neighbors
VI_ITERS = 24             # value-iteration sweeps per tick
R_STEP = -1.0             # time penalty each step to encourage finishing
EXIT_REWARD = 300.0       # terminal reward on exit
HAZARD_PENALTY = -2000.0  # explosion/monster/bomb/adjacent/next-step tiles
WALL_PENALTY = -1e6       # impossible

# Risk map (where monsters likely will be soon)
RISK_HORIZON = 4
RISK_DECAY = 0.65
RISK_SCALE = 400.0        # bigger => more enemy avoidance
RISK_RADIUS_EXP = 0.6     # soften sharp spikes

# Distance-to-exit shaping
EXIT_SHAPING = 35.0


def in_bounds(x, y, wrld):
    return 0 <= x < wrld.width() and 0 <= y < wrld.height()

def is_exit(x, y, wrld):
    """True if (x,y) is the exit cell (and in bounds)."""
    return in_bounds(x, y, wrld) and wrld.exit_at(x, y)


def neighbors8(x, y, wrld):
    for dx, dy in DIRS8:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, wrld) and not wrld.wall_at(nx, ny):
            yield nx, ny


def flood_to_exit(wrld):
    """8-connected backward BFS distances to exit cells."""
    dist = [[INF] * wrld.height() for _ in range(wrld.width())]
    q = deque()
    for y in range(wrld.height()):
        for x in range(wrld.width()):
            if wrld.exit_at(x, y):
                dist[x][y] = 0
                q.append((x, y))
    while q:
        cx, cy = q.popleft()
        d0 = dist[cx][cy]
        for nx, ny in neighbors8(cx, cy, wrld):
            if dist[nx][ny] > d0 + 1:
                dist[nx][ny] = d0 + 1
                q.append((nx, ny))
    return dist


def min_cheby_to_monster(x, y, wrld, radius=2):
    """Minimum Chebyshev distance to any monster (<= radius) or INF if none within radius."""
    best = INF
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, wrld):
                continue
            if wrld.monsters_at(nx, ny):
                d = max(abs(dx), abs(dy))
                if d < best:
                    best = d
    return best


def monster_reachable_next_step(x, y, wrld):
    """True if any monster can step into (x,y) in one tick (including staying)."""
    if wrld.monsters_at(x, y):
        return True
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            sx, sy = x + dx, y + dy
            if in_bounds(sx, sy, wrld) and wrld.monsters_at(sx, sy):
                if not wrld.wall_at(x, y):
                    return True
    return False


def immediate_hazard(x, y, wrld):
    """
    Explosion/monster/bomb present now, OR adjacent to a monster,
    OR a tile a monster can reach next step.
    """
    if wrld.explosion_at(x, y):
        return True
    if wrld.bomb_at(x, y) is not None:
        return True
    if wrld.monsters_at(x, y):
        return True
    if monster_reachable_next_step(x, y, wrld):
        return True
    if min_cheby_to_monster(x, y, wrld, radius=1) <= 1:
        return True
    return False


def build_risk_map(wrld):
    """Diffuse monster presence a few steps ahead to get a probabilistic danger field."""
    W, H = wrld.width(), wrld.height()
    risk = [[0.0] * H for _ in range(W)]

    # Seed with current monster positions
    seed = [[0.0] * H for _ in range(W)]
    for y in range(H):
        for x in range(W):
            if wrld.monsters_at(x, y):
                seed[x][y] += 1.0
    total = sum(seed[x][y] for x in range(W) for y in range(H))
    if total > 0:
        for x in range(W):
            for y in range(H):
                seed[x][y] /= total

    # Bombs add baseline risk
    for y in range(H):
        for x in range(W):
            if wrld.bomb_at(x, y) is not None:
                risk[x][y] += 0.5

    current = seed
    decay = 1.0
    for _ in range(RISK_HORIZON):
        decay *= RISK_DECAY
        # accumulate softened risk
        for x in range(W):
            for y in range(H):
                p = current[x][y]
                if p > 0:
                    risk[x][y] += decay * (p ** RISK_RADIUS_EXP)

        # spread to neighbors (monster can stay or move to any walkable neighbor)
        nxt = [[0.0] * H for _ in range(W)]
        for x in range(W):
            for y in range(H):
                p = current[x][y]
                if p <= 0:
                    continue
                opts = [(x, y)] + list(neighbors8(x, y, wrld))
                share = p / len(opts)
                for (nx, ny) in opts:
                    nxt[nx][ny] += share
        current = nxt

    # clamp
    for x in range(W):
        for y in range(H):
            risk[x][y] = min(1.0, max(0.0, risk[x][y]))

    return risk


def mdp_value_iteration(wrld, dist_exit, risk):
    """Solve grid MDP; return greedy policy mapping (x,y)->(dx,dy)."""
    W, H = wrld.width(), wrld.height()

    # Precompute legal options per state
    legal = {}
    for x in range(W):
        for y in range(H):
            if wrld.wall_at(x, y):
                continue
            opts = [(x, y)]  # allow stay
            opts += list(neighbors8(x, y, wrld))
            legal[(x, y)] = opts

    def R(x, y):
        if wrld.wall_at(x, y):
            return WALL_PENALTY
        if immediate_hazard(x, y, wrld):
            return HAZARD_PENALTY
        r = R_STEP
        d = dist_exit[x][y]
        if d < INF:
            r += EXIT_SHAPING * (1.0 / (1.0 + d))
        r -= RISK_SCALE * risk[x][y]
        return r

    # init values; exits are terminal with EXIT_REWARD
    V = [[0.0] * H for _ in range(W)]
    terminal = [[False] * H for _ in range(W)]
    for y in range(H):
        for x in range(W):
            if wrld.exit_at(x, y):
                V[x][y] = EXIT_REWARD
                terminal[x][y] = True

    # value iteration
    for _ in range(VI_ITERS):
        for y in range(H):
            for x in range(W):
                if wrld.wall_at(x, y) or terminal[x][y]:
                    continue
                best_q = -INF
                my_opts = legal[(x, y)]
                for ax, ay in DIRS9:
                    tx, ty = x + ax, y + ay
                    if not in_bounds(tx, ty, wrld) or wrld.wall_at(tx, ty):
                        tx, ty = x, y  # bump into wall -> stay

                    q = 0.0
                    # intended
                    q += P_INTEND * (R(tx, ty) + DISCOUNT * V[tx][ty])

                    # slips to any other legal spot
                    others = [s for s in my_opts if s != (tx, ty)]
                    if others:
                        ps = (1.0 - P_INTEND) / len(others)
                        for (sx, sy) in others:
                            q += ps * (R(sx, sy) + DISCOUNT * V[sx][sy])

                    if q > best_q:
                        best_q = q
                V[x][y] = best_q

    # greedy policy extraction
    Pi = {}
    for y in range(H):
        for x in range(W):
            if wrld.wall_at(x, y):
                continue
            if terminal[x][y]:
                Pi[(x, y)] = (0, 0)
                continue

            best_q = -INF
            best_a = (0, 0)
            my_opts = legal[(x, y)]
            for ax, ay in DIRS9:
                tx, ty = x + ax, y + ay
                if not in_bounds(tx, ty, wrld) or wrld.wall_at(tx, ty):
                    tx, ty = x, y

                q = P_INTEND * (R(tx, ty) + DISCOUNT * V[tx][ty])
                others = [s for s in my_opts if s != (tx, ty)]
                if others:
                    ps = (1.0 - P_INTEND) / len(others)
                    for (sx, sy) in others:
                        q += ps * (R(sx, sy) + DISCOUNT * V[sx][sy])
                if q > best_q:
                    best_q = q
                    best_a = (ax, ay)
            Pi[(x, y)] = best_a

    return Pi


class TestCharacter(CharacterEntity):

    def __init__(self, name="hero", avatar="C", *args, **kwargs):
        super().__init__(name, avatar, *args, **kwargs)

    def do(self, wrld):
        me = wrld.me(self)
        if me is None:
            return

        # distance shaping
        dist_exit = flood_to_exit(wrld)

        # danger field
        risk = build_risk_map(wrld)

        # MDP solve
        Pi = mdp_value_iteration(wrld, dist_exit, risk)

        # 4) proposed action
        ax, ay = Pi.get((me.x, me.y), (0, 0))
        tx, ty = me.x + ax, me.y + ay

        # min fix for exit issue bug
        if is_exit(tx, ty, wrld):
            self.move(ax, ay)
            return
        # If policy didn't pick it but exit is adjacent, take it anyway
        for dx, dy in DIRS9:
            nx, ny = me.x + dx, me.y + dy
            if is_exit(nx, ny, wrld):
                self.move(dx, dy)
                return


        # safety net
        def safe_score(x, y):
            # Prefer lower risk and closer to exit; forbid immediate hazards.
            if not in_bounds(x, y, wrld) or wrld.wall_at(x, y):
                return -INF
            if immediate_hazard(x, y, wrld):
                return -INF / 2  # absolute no-go
            d = dist_exit[x][y]
            closeness = (100.0 / (1.0 + d)) if d < INF else -100.0
            near = min_cheby_to_monster(x, y, wrld, radius=2)
            prox_pen = 0.0 if near == INF else (25.0 / (1.0 + near))
            return closeness - (RISK_SCALE * risk[x][y]) - prox_pen

        unsafe = (
            not in_bounds(tx, ty, wrld) or
            wrld.wall_at(tx, ty) or
            immediate_hazard(tx, ty, wrld)
        )

        if unsafe:
            best = (0, 0)
            best_s = safe_score(me.x, me.y)  # sometimes staying is safest
            for dx, dy in DIRS9:
                nx, ny = me.x + dx, me.y + dy
                s = safe_score(nx, ny)
                if s > best_s:
                    best_s = s
                    best = (dx, dy)
            ax, ay = best
        # ------------------------------------------------------

        # visualization: exit stripes (blue/cyan), risky cells (red)
        self._draw_overlay(wrld, dist_exit, risk)

        self.move(ax, ay)

    # ---------- visualization ----------
    def _draw_overlay(self, wrld, dist_exit, risk):
        W, H = wrld.width(), wrld.height()
        for x in range(W):
            for y in range(H):
                if wrld.wall_at(x, y):
                    continue
                base = Back.BLUE if (dist_exit[x][y] < INF and dist_exit[x][y] % 2 == 0) else Back.CYAN
                if dist_exit[x][y] >= INF:
                    base = Back.BLACK
                if risk[x][y] > 0.25:
                    self.set_cell_color(x, y, Fore.WHITE + Back.RED)
                else:
                    self.set_cell_color(x, y, Fore.WHITE + base)
