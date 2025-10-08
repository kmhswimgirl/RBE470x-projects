# This is necessary to find the main code
import sys
import math
import random

sys.path.insert(0, '../Bomberman')
from entity import CharacterEntity  # type: ignore
from colorama import Fore, Back
from collections import deque

# --- Constants & Parameters ---

INF = 10**9
DIRS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
DIRS9 = [(0, 0)] + DIRS8

# MDP params
DISCOUNT = 0.95
P_INTEND = 0.80
VI_ITERS = 24
R_STEP = -1.0
EXIT_REWARD = 300.0
HAZARD_PENALTY = -2000.0
WALL_PENALTY = -1e6

# Risk map params
RISK_HORIZON = 4
RISK_DECAY = 0.65
RISK_SCALE = 400.0
RISK_RADIUS_EXP = 0.6

# Distance-to-exit shaping
EXIT_SHAPING = 35.0


# --- Helper Functions ---

def in_bounds(x, y, wrld):
    return 0 <= x < wrld.width() and 0 <= y < wrld.height()


def is_exit(x, y, wrld):
    return in_bounds(x, y, wrld) and wrld.exit_at(x, y)


def neighbors8(x, y, wrld):
    for dx, dy in DIRS8:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, wrld) and not wrld.wall_at(nx, ny):
            yield nx, ny


def flood_to_exit(wrld):
    """Compute 8-connected backward BFS distances to exits."""
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

    for y in range(H):
        for x in range(W):
            if wrld.bomb_at(x, y) is not None:
                risk[x][y] += 0.5

    current = seed
    decay = 1.0
    for _ in range(RISK_HORIZON):
        decay *= RISK_DECAY
        for x in range(W):
            for y in range(H):
                p = current[x][y]
                if p > 0:
                    risk[x][y] += decay * (p ** RISK_RADIUS_EXP)

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

    for x in range(W):
        for y in range(H):
            risk[x][y] = min(1.0, max(0.0, risk[x][y]))

    return risk


def mdp_value_iteration(wrld, dist_exit, risk):
    """Solve grid MDP; return greedy policy mapping (x,y)->(dx,dy)."""
    W, H = wrld.width(), wrld.height()
    legal = {}
    for x in range(W):
        for y in range(H):
            if wrld.wall_at(x, y):
                continue
            opts = [(x, y)] + list(neighbors8(x, y, wrld))
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

    V = [[0.0] * H for _ in range(W)]
    terminal = [[False] * H for _ in range(W)]
    for y in range(H):
        for x in range(W):
            if wrld.exit_at(x, y):
                V[x][y] = EXIT_REWARD
                terminal[x][y] = True

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
                        tx, ty = x, y

                    q = P_INTEND * (R(tx, ty) + DISCOUNT * V[tx][ty])
                    others = [s for s in my_opts if s != (tx, ty)]
                    if others:
                        ps = (1.0 - P_INTEND) / len(others)
                        for (sx, sy) in others:
                            q += ps * (R(sx, sy) + DISCOUNT * V[sx][sy])
                    if q > best_q:
                        best_q = q
                V[x][y] = best_q

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


# --- Main Character Class ---

class TestCharacter(CharacterEntity):
    def __init__(self, name="hero", avatar="C", *args, **kwargs):
        super().__init__(name, avatar, *args, **kwargs)
        self._bomb_cooldown = 0
        self._BOMB_COOLDOWN_TICKS = 10
        self._bomb_active = False
        self._exit_goal = None

    def do(self, wrld):
        me = wrld.me(self)
        if me is None:
            return

        dist_exit = flood_to_exit(wrld)
        risk = build_risk_map(wrld)
        Pi = mdp_value_iteration(wrld, dist_exit, risk)

        # Find nearest exit coordinate
        exit_pos = None
        best_exit_d = INF
        for y in range(wrld.height()):
            for x in range(wrld.width()):
                if wrld.exit_at(x, y):
                    d = math.hypot(x - me.x, y - me.y)
                    if d < best_exit_d:
                        best_exit_d = d
                        exit_pos = (x, y)
        self._exit_goal = exit_pos

        # Track if a bomb exists in world
        self._bomb_active = any(
            wrld.bomb_at(x, y) for x in range(wrld.width()) for y in range(wrld.height())
        )

        # --- Define safety scoring ---
        def safe_score(x, y):
            if not in_bounds(x, y, wrld) or wrld.wall_at(x, y):
                return -INF
            if immediate_hazard(x, y, wrld):
                return -INF / 2
            d = dist_exit[x][y]
            closeness = (100.0 / (1.0 + d)) if d < INF else -100.0
            near = min_cheby_to_monster(x, y, wrld, radius=2)
            prox_pen = 0.0 if near == INF else (25.0 / (1.0 + near))
            return closeness - (RISK_SCALE * risk[x][y]) - prox_pen

        # --- Behavior when exit unreachable ---
        unreachable = dist_exit[me.x][me.y] == INF
        if unreachable and not self._bomb_active:
            if self._exit_goal is not None:
                gx, gy = self._exit_goal
                dx = int(math.copysign(1, gx - me.x)) if gx != me.x else 0
                dy = int(math.copysign(1, gy - me.y)) if gy != me.y else 0
                tx, ty = me.x + dx, me.y + dy

                # Wall ahead → bomb it
                if in_bounds(tx, ty, wrld) and wrld.wall_at(tx, ty) and self._bomb_cooldown <= 0:
                    self.place_bomb()
                    self._bomb_cooldown = self._BOMB_COOLDOWN_TICKS
                    self._bomb_active = True
                    return

                # Move toward goal direction
                if in_bounds(tx, ty, wrld) and not wrld.wall_at(tx, ty) and not immediate_hazard(tx, ty, wrld):
                    self.move(dx, dy)
                    return

                # Otherwise, pick safest nearby spot
                best = (0, 0)
                best_s = -INF
                for sx, sy in DIRS8:
                    nx, ny = me.x + sx, me.y + sy
                    if in_bounds(nx, ny, wrld) and not wrld.wall_at(nx, ny):
                        s = safe_score(nx, ny)
                        if s > best_s:
                            best_s = s
                            best = (sx, sy)
                self.move(*best)
                return

        # --- Normal movement (exit reachable or bomb active) ---
        ax, ay = Pi.get((me.x, me.y), (0, 0))
        tx, ty = me.x + ax, me.y + ay

        if is_exit(tx, ty, wrld):
            self.move(ax, ay)
            return
        for dx, dy in DIRS9:
            nx, ny = me.x + dx, me.y + dy
            if is_exit(nx, ny, wrld):
                self.move(dx, dy)
                return

        unsafe = (
            not in_bounds(tx, ty, wrld)
            or wrld.wall_at(tx, ty)
            or immediate_hazard(tx, ty, wrld)
        )
        if unsafe:
            best = (0, 0)
            best_s = safe_score(me.x, me.y)
            for dx, dy in DIRS9:
                nx, ny = me.x + dx, me.y + dy
                s = safe_score(nx, ny)
                if s > best_s:
                    best_s = s
                    best = (dx, dy)
            ax, ay = best

        self._draw_overlay(wrld, dist_exit, risk)
        self.move(ax, ay)

        if self._bomb_cooldown > 0:
            self._bomb_cooldown -= 1

        # Reset bomb state when none are left
        if self._bomb_active:
            bombs_left = any(
                wrld.bomb_at(x, y) for x in range(wrld.width()) for y in range(wrld.height())
            )
            if not bombs_left:
                self._bomb_active = False

    # --- Visualization ---
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
