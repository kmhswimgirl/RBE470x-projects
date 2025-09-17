# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')

from entity import CharacterEntity
from colorama import Fore, Back
import math, random

class TestCharacter2(CharacterEntity):
    # ----- Search toggle -----
    SEARCH = 'sa'   # 'sa' for Simulated Annealing, 'shc' for Stochastic Hill Climb

    # ----- SA params -----
    SA_INIT_TEMP = 3.0
    SA_COOL      = 0.85
    SA_ITERS     = 20

    # ----- SHC params -----
    SHC_PROB_RANDOM = 0.2

    # ----- Cost weights -----
    W_DIST      = 1.0     # distance to exit
    W_SOFT_DANG = 1.0     # soft-danger (monster's neighbor) additive penalty
    W_BACKTRACK = 0.3     # discourage immediate backtrack
    W_RISK      = 6.0     # weight for probabilistic collision risk (sum over monsters)

    # ----- Safety gates -----
    INF_HARD     = 10_000      # forbid stepping directly onto a monster
    RISK_MAX     = 0.35        # disallow candidates with risk >= this (unless stepping onto exit)
    PANIC_RANGE  = 1           # if any monster within Chebyshev distance <= this -> panic mode

    def do(self, wrld):
        me = wrld.me(self)
        start = (me.x, me.y)

        if not hasattr(self, "prev"):
            self.prev = None

        # Already won?
        if wrld.exit_at(start[0], start[1]):
            self.move(0, 0)
            return

        # Find exit
        goal = None
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    goal = (x, y); break
            if goal: break
        if not goal:
            self.move(0, 0); return

        # If the exit is adjacent, take it immediately
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                tx, ty = start[0]+dx, start[1]+dy
                if 0 <= tx < wrld.width() and 0 <= ty < wrld.height():
                    if wrld.exit_at(tx, ty):
                        self.prev = start
                        self.move(dx, dy)
                        return

        # Helpers
        def inb(x, y): return 0 <= x < wrld.width() and 0 <= y < wrld.height()
        def passable(x, y):
            return (inb(x, y)
                    and not wrld.wall_at(x, y)
                    and not wrld.bomb_at(x, y)
                    and not wrld.explosion_at(x, y))
        def cheb(a, b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

        # Gather monsters and their legal next cells
        monsters = []
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                ms = wrld.monsters_at(x, y)
                if ms:
                    # There can be multiple in a cell; treat each independently
                    for _m in ms:
                        monsters.append((x, y))

        hard_danger = set((mx, my) for (mx, my) in monsters)

        # Precompute each monster's neighbor set and branching factor
        mon_neighbors = []   # list of sets
        min_dist_to_mon = float('inf')
        for (mx, my) in monsters:
            nbrs = set()
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx==0 and dy==0: continue
                    nx, ny = mx+dx, my+dy
                    if passable(nx, ny):
                        nbrs.add((nx, ny))
            mon_neighbors.append(nbrs)
            # track distance to closest monster for panic mode
            min_dist_to_mon = min(min_dist_to_mon, cheb((mx, my), start))

        # Candidate moves (8-neighborhood, passable, not directly on a monster)
        candidates = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                nx, ny = start[0]+dx, start[1]+dy
                if passable(nx, ny) and (nx, ny) not in hard_danger:
                    candidates.append((nx, ny))
        if not candidates:
            self.move(0, 0); return

        # Risk model: probability a monster lands on 'cell' next tick
        # = sum over monsters of 1/deg(monster) if cell in its neighbor set; clamp to 1
        def next_step_risk(cell):
            risk = 0.0
            for nbrs in mon_neighbors:
                deg = len(nbrs)
                if (deg > 0) and (cell in nbrs):
                    risk += 1.0 / deg
                # If deg == 0, monster is stuck; ignore.
            return min(risk, 1.0)

        # Soft danger indicator (monster could move there)
        soft_danger = set().union(*mon_neighbors) if mon_neighbors else set()

        # Cost for SA/SHC
        def cost(cell):
            if cell in hard_danger:
                return self.INF_HARD
            d = self.W_DIST * cheb(cell, goal)
            if cell in soft_danger:
                d += self.W_SOFT_DANG
            if self.prev == cell:
                d += self.W_BACKTRACK
            # probabilistic collision risk penalty
            r = next_step_risk(cell)
            d += self.W_RISK * r
            return d

        # Safety-gated candidate list (still allow exit even if risky)
        gated = []
        for c in candidates:
            r = next_step_risk(c)
            if r >= self.RISK_MAX and not wrld.exit_at(c[0], c[1]):
                continue
            gated.append(c)

        # If everything is too risky, fall back to the least risky among all candidates
        if not gated:
            gated = sorted(candidates, key=lambda c: (next_step_risk(c), cost(c)))[:3]

        # Debug coloring
        for (cx, cy) in candidates:
            if (cx, cy) in hard_danger:
                self.set_cell_color(cx, cy, Fore.WHITE + Back.RED)
            elif wrld.exit_at(cx, cy):
                self.set_cell_color(cx, cy, Fore.WHITE + Back.BLUE)
            else:
                r = next_step_risk((cx, cy))
                if r >= self.RISK_MAX:
                    self.set_cell_color(cx, cy, Fore.BLACK + Back.MAGENTA)  # too risky (gated out)
                elif (cx, cy) in soft_danger:
                    self.set_cell_color(cx, cy, Fore.BLACK + Back.YELLOW)   # soft danger but allowed
                else:
                    self.set_cell_color(cx, cy, Fore.BLACK + Back.GREEN)    # safe
        self.set_cell_color(goal[0], goal[1], Fore.WHITE + Back.BLUE)

        # Panic mode: monster is adjacent → lower temperature and choose lowest-risk move
        panic = (min_dist_to_mon <= self.PANIC_RANGE)

        pick = None
        if self.SEARCH == 'sa' and not panic:
            rng = random.random
            # Anneal on gated candidates
            current = min(gated, key=cost)
            current_cost = cost(current)
            best = current; best_cost = current_cost
            T = self.SA_INIT_TEMP
            for _ in range(self.SA_ITERS):
                nxt = gated[math.floor(rng()*len(gated))]
                nxt_cost = cost(nxt)
                delta = nxt_cost - current_cost
                if delta <= 0 or rng() < math.exp(-delta / max(T, 1e-6)):
                    current, current_cost = nxt, nxt_cost
                    if current_cost < best_cost:
                        best, best_cost = current, current_cost
                T *= self.SA_COOL
            pick = best
        else:
            # Panic or SHC: minimize cost with a pinch of randomness among improving moves
            here_cost = cost(start)
            scored = sorted(((c, cost(c)) for c in gated), key=lambda t: t[1])
            improving = [c for (c, cst) in scored if cst < here_cost]
            if self.SEARCH == 'shc' and improving and random.random() < self.SHC_PROB_RANDOM:
                pick = random.choice(improving)
            else:
                pick = scored[0][0]

        # Move
        if pick:
            self.prev = start
            self.move(pick[0] - start[0], pick[1] - start[1])
        else:
            self.move(0, 0)
