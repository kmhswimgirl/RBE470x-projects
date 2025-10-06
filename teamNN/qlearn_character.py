# Approximate Q-Learning Bomberman Agent (Integrated Version)
# ------------------------------------------------------------
# Integrates directly with the Bomberman engine as the professor's variants do.
# Loads world from map file (hardcoded, no fallback).
# Includes rewards for reaching exit and destroying walls.
# ------------------------------------------------------------

from collections import defaultdict, deque
import random
import os
import json
import argparse
import sys

# -------- Engine Imports -------- #
_here = os.path.dirname(os.path.abspath(__file__))
_bomberman_dir = os.path.abspath(os.path.join(_here, '..', 'Bomberman'))
_project_root = os.path.abspath(os.path.join(_here, '..'))
if _bomberman_dir not in sys.path:
    sys.path.insert(0, _bomberman_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Bomberman.game import Game 
from Bomberman.world import World
from Bomberman.entity import CharacterEntity

# ---------------------------- Hyperparameters ---------------------------- #
ALPHA = 0.12
GAMMA = 0.95
EPSILON = 0.15

RISK_HORIZON = 3
RISK_DECAY = 0.7
RISK_RADIUS_EXP = 0.7

R_STEP = -1.0
R_EXIT = 500.0
R_DEATH = -1200.0
R_BOMB_CLEAR_PATH = +30.0
R_BOMB_WALL_DESTROYED = +40.0

DIRS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
DIRS9 = [(0, 0)] + DIRS8
INF = 10 ** 9

# ---------------------------- Utilities ---------------------------- #

def in_bounds(x, y, wrld):
    return 0 <= x < wrld.width() and 0 <= y < wrld.height()

def neighbors8(x, y, wrld):
    for dx, dy in DIRS8:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, wrld) and not wrld.wall_at(nx, ny):
            yield nx, ny

def flood_to_exit(wrld):
    """8-connected BFS distance to nearest exit."""
    W, H = wrld.width(), wrld.height()
    dist = [[INF] * H for _ in range(W)]
    q = deque()
    for y in range(H):
        for x in range(W):
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

def path_exists_to_exit(wrld, startx, starty):
    seen = set()
    q = deque([(startx, starty)])
    while q:
        x, y = q.popleft()
        if wrld.exit_at(x, y):
            return True
        for nx, ny in neighbors8(x, y, wrld):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False

def immediate_hazard(x, y, wrld):
    if wrld.explosion_at(x, y): return True
    if wrld.bomb_at(x, y) is not None: return True
    if wrld.monsters_at(x, y): return True
    return False

def build_risk_map(wrld) -> list[list[float]]:
    """Diffuse monster presence a few steps ahead.
    
    Parameters:
        wrld: World object

    Returns:
        2D list of floats representing risk levels at each cell

    """
    W, H = wrld.width(), wrld.height()
    risk = [[0.0] * H for _ in range(W)]
    seed = [[0.0] * H for _ in range(W)]
    for y in range(H):
        for x in range(W):
            if wrld.monsters_at(x, y):
                seed[x][y] = 1.0
    total = sum(seed[x][y] for x in range(W) for y in range(H))
    if total > 0:
        for x in range(W):
            for y in range(H):
                seed[x][y] /= total
    current = seed
    decay = 1.0
    for _ in range(RISK_HORIZON):
        decay *= RISK_DECAY
        nxt = [[0.0] * H for _ in range(W)]
        for x in range(W):
            for y in range(H):
                p = current[x][y]
                if p > 0:
                    risk[x][y] += decay * (p ** RISK_RADIUS_EXP)
                    for nx, ny in neighbors8(x, y, wrld):
                        nxt[nx][ny] += p / 8.0
        current = nxt
    return risk

# ---------------------------- Q-Learning Agent ---------------------------- #

class ApproxQLearningCharacter(CharacterEntity):
    def __init__(self, name="qhero", avatar="Q", x=0, y=0, *args, **kwargs):
        super().__init__(name, avatar, x, y, *args, **kwargs)
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.weights = defaultdict(float)
        self.prev_wrld = None
        self.prev_features = None
        self.prev_action = None
        self._dist_exit = None
        self._risk = None

    def do(self, wrld):
        me = wrld.me(self)
        if me is None:
            return

        self._dist_exit = flood_to_exit(wrld)
        self._risk = build_risk_map(wrld)

        if self.prev_wrld and self.prev_features is not None and self.prev_action is not None:
            reward = self._compute_reward(self.prev_wrld, wrld, self.prev_action)
            td_error = (reward + self.gamma * self._max_q(wrld)) - self._q_from_features(self.prev_features)
            for f, val in self.prev_features.items():
                self.weights[f] += self.alpha * td_error * val

        action = self._choose_action(wrld)
        self.prev_wrld = wrld
        self.prev_action = action
        self.prev_features = self._extract_features(wrld, action)

        atype, dx, dy = action
        if atype == "bomb":
            self.place_bomb()
        else:
            self.move(dx, dy)

    # -------- Q helpers -------- #
    def _actions(self, wrld):
        me = wrld.me(self)
        acts = [("move", dx, dy) for dx, dy in DIRS9 if in_bounds(me.x + dx, me.y + dy, wrld)]
        if self._should_offer_bomb(wrld):
            acts.append(("bomb", 0, 0))
        return acts

    def _choose_action(self, wrld):
        acts = self._actions(wrld)
        if not acts:
            return ("move", 0, 0)
        if random.random() < self.epsilon:
            return random.choice(acts)
        return max(acts, key=lambda a: self._q_value(wrld, a))

    def _q_value(self, wrld, action):
        return self._q_from_features(self._extract_features(wrld, action))

    def _q_from_features(self, feats):
        return sum(self.weights[f] * v for f, v in feats.items())

    def _max_q(self, wrld):
        acts = self._actions(wrld)
        return max((self._q_value(wrld, a) for a in acts), default=0.0)

    # -------- Features -------- #
    def _extract_features(self, wrld, action):
        atype, ax, ay = action
        me = wrld.me(self)
        feats = defaultdict(float)
        tx, ty = me.x + ax, me.y + ay
        if not in_bounds(tx, ty, wrld):
            tx, ty = me.x, me.y
        feats["bias"] = 1.0
        if self._dist_exit is None:
            self._dist_exit = flood_to_exit(wrld)
        d = self._dist_exit[tx][ty]
        feats["inv_exit_dist"] = 1.0 / (1.0 + (d if d < INF else 9999))
        if self._risk is None:
            self._risk = build_risk_map(wrld)
        feats["risk_here"] = self._risk[tx][ty]
        feats["blocked_room"] = 1.0 if not path_exists_to_exit(wrld, me.x, me.y) else 0.0
        feats["act_is_bomb"] = 1.0 if atype == "bomb" else 0.0
        return feats

    # -------- Reward -------- #
    def _compute_reward(self, prev_wrld, curr_wrld, prev_action):
        me_prev = prev_wrld.me(self)
        me_curr = curr_wrld.me(self)
        if me_prev and me_curr is None:
            return R_DEATH
        if me_curr and curr_wrld.exit_at(me_curr.x, me_curr.y):
            return R_EXIT

        reward = R_STEP
        atype, _, _ = prev_action

        # Bomb cleared path
        if atype == "bomb" and not path_exists_to_exit(prev_wrld, me_prev.x, me_prev.y):
            if path_exists_to_exit(curr_wrld, me_prev.x, me_prev.y):
                reward += R_BOMB_CLEAR_PATH

        # Bomb destroyed walls
        walls_before = sum(prev_wrld.wall_at(x, y) for x in range(prev_wrld.width()) for y in range(prev_wrld.height()))
        walls_after = sum(curr_wrld.wall_at(x, y) for x in range(curr_wrld.width()) for y in range(curr_wrld.height()))
        if walls_after < walls_before:
            reward += R_BOMB_WALL_DESTROYED * (walls_before - walls_after)

        return reward

    def _should_offer_bomb(self, wrld):
        me = wrld.me(self)
        if immediate_hazard(me.x, me.y, wrld):
            return False
        if not path_exists_to_exit(wrld, me.x, me.y):
            return True
        return False

    # -------- Persistence -------- #
    def save_weights(self, path="weights.json"):
        with open(path, "w") as f:
            json.dump(self.weights, f)

    def load_weights(self, path="weights.json"):
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
                self.weights = defaultdict(float, {k: float(v) for k, v in d.items()})

# ---------------------------- Training Harness ---------------------------- #

def train_agent(episodes=20, map_file="map.txt", weight_file="weights.json"):
    agent = ApproxQLearningCharacter()
    if os.path.exists(weight_file):
        agent.load_weights(weight_file)

    # Training loop
    for ep in range(episodes):
        # Initialize game and world
        gm = Game.fromfile(map_file)
        wrld = World()

        # Place agent on first empty cell
        placed = False
        for y in range(wrld.height()):
            for x in range(wrld.width()):
                if wrld.empty_at(x, y):
                    agent.x, agent.y = x, y
                    gm.add_character(agent)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            gm.add_character(agent)

        # Run episode
        for t in range(200):
            agent.do(wrld)
            wrld.next()
        print(f"Episode {ep+1}/{episodes} complete")

    agent.save_weights(weight_file)
    print(f"Training finished, weights saved to {weight_file}")

# ---------------------------- Main Entry ---------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, help="Number of training episodes", default=0)
    parser.add_argument("--map", type=str, help="Map file to load", default='map.txt')
    parser.add_argument("--play", action="store_true", help="Run one play using saved weights")
    args = parser.parse_args()

    if args.train > 0:
        train_agent(args.train, args.map)
    elif args.play:
        print("Play mode: load weights and run greedy policy.")
        agent = ApproxQLearningCharacter()
        agent.load_weights()
        gm = Game.fromfile(args.map)
        wrld = World()
        gm.add_character(agent)
        gm.go()
    else:
        print("Specify --train N or --play")
