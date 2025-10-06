# Approximate Q-Learning Bomberman Character
# ------------------------------------------
# Works seamlessly in variants:
#   - TRAINING = True  → updates weights after each Game.go()
#   - TRAINING = False → runs using saved weights only
# Character automatically saves/loads "weights.json" between games.

import os, json, random
from collections import defaultdict, deque
from entity import CharacterEntity  # from Bomberman engine

# ---------------- Configuration ---------------- #
TRAINING = True          # <-- Flip this to False when fully trained
ALPHA = 0.1              # learning rate
GAMMA = 0.95              # discount factor
EPSILON = 0.15           # exploration probability during training
WEIGHT_FILE = "weights.json"

R_EXIT = +500.0 # Reward for reaching exit
R_DEATH = -1200.0 # Penalty for dying
R_STEP = -1.0 # Small penalty for each step taken
R_BOMB_WALL = +50.0 # Reward for bombing a wall
R_CLEAR_PATH = +200.0 # Reward for clearing a path
R_MONSTER_KILL = +300.0 # Reward for killing a monster
R_BOMB_CLEAR_PATH = +100.0 # Reward for increasing reachable cells

DIRS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
DIRS9 = [(0, 0)] + DIRS8
INF = 10 ** 9

# ---------------- Helper functions ---------------- #

def in_bounds(x, y, wrld):
    return 0 <= x < wrld.width() and 0 <= y < wrld.height()

def neighbors8(x, y, wrld):
    for dx, dy in DIRS8:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, wrld) and not wrld.wall_at(nx, ny):
            yield nx, ny

def flood_to_exit(wrld):
    """
    Compute the distance from each cell to the nearest exit using multi-source BFS.
    Parameters:
    - wrld: The game world object.
    Returns:
    - A 2D list of distances where dist[x][y] is the distance from (x, y) to the nearest exit.
      If a cell is unreachable, its distance is INF.

    """


    W, H = wrld.width(), wrld.height()
    dist = [[INF]*H for _ in range(W)]
    q = deque()
    for y in range(H):
        for x in range(W):
            if wrld.exit_at(x, y):
                dist[x][y] = 0
                q.append((x, y))
    while q:
        cx, cy = q.popleft()
        for nx, ny in neighbors8(cx, cy, wrld):
            if dist[nx][ny] > dist[cx][cy] + 1:
                dist[nx][ny] = dist[cx][cy] + 1
                q.append((nx, ny))
    return dist

def path_exists_to_exit(wrld, sx, sy):
    """
    Check if there is a path from (sx, sy) to any exit in the world.

    Parameters:
    - wrld: The game world object.
    - sx, sy: Starting coordinates.

    Returns:
    - True if a path exists to an exit, False otherwise.
    """
    seen, q = set(), deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        if wrld.exit_at(x, y): return True
        for nx, ny in neighbors8(x, y, wrld):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False

def reachable_cells(sx, sy, wrld):
    """
    Find the number of cells reachable from (sx, sy) without crossing walls.
    Parameters:
    - sx, sy: Starting coordinates.
    - wrld: The game world object.

    Returns:
    - number of total reachable cells
    """
    seen, q = set(), deque([(sx, sy)])
    seen.add((sx, sy))
    while q:
        x, y = q.popleft()
        for nx, ny in neighbors8(x, y, wrld):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return len(seen)

def immediate_hazard(x, y, wrld):
    if wrld.explosion_at(x, y): return True
    if wrld.bomb_at(x, y): return True
    if wrld.monsters_at(x, y): return True
    return False

# ---------------- Q-Learning Character ---------------- #

class ApproxQLearningCharacter(CharacterEntity):
    def __init__(self, name="qhero", avatar="Q", x=0, y=0):
        super().__init__(name, avatar, x, y)
        self.alpha, self.gamma, self.epsilon = ALPHA, GAMMA, EPSILON
        self.weights = defaultdict(float)
        if os.path.exists(WEIGHT_FILE):
            self.load_weights()
            print(f"[QL] Loaded existing weights from {WEIGHT_FILE}")
        self.prev_features = None
        self.prev_action = None
        self.prev_wrld = None
        self._dist_exit = 1000, 1000

    def do(self, wrld):
        """Main loop called every tick by Game.go()."""
        me = wrld.me(self)
        if me is None:
            return

        # Compute distances, features
        self._dist_exit = flood_to_exit(wrld)

        # Q-learning update (only during training)
        if TRAINING and self.prev_features and self.prev_action and self.prev_wrld:
            reward = self._compute_reward(self.prev_wrld, wrld, self.prev_action)
            td_error = (reward + self.gamma * self._max_q(wrld)) - self._q_from_features(self.prev_features)
            for f, v in self.prev_features.items():
                self.weights[f] += self.alpha * td_error * v

        # Choose next action
        action = self._choose_action(wrld)
        self.prev_wrld, self.prev_action = wrld, action
        self.prev_features = self._extract_features(wrld, action)

        # Execute action
        atype, dx, dy = action
        if atype == "bomb": self.place_bomb()
        else: self.move(dx, dy)

        # If training and the game ends (agent dies or reaches exit), save weights
        if TRAINING:
            me = wrld.me(self)
            if me is None or wrld.exit_at(self.x, self.y):
                self.save_weights()

    # ---------------- Core Q-Learning ---------------- #
    def _actions(self, wrld):
        me = wrld.me(self)
        acts = [("move", dx, dy) for dx, dy in DIRS9 if in_bounds(me.x + dx, me.y + dy, wrld)]
        if self._should_offer_bomb(wrld):
            acts.append(("bomb", 0, 0))
        return acts

    def _choose_action(self, wrld):
        acts = self._actions(wrld)
        if not acts: return ("move", 0, 0)
        if TRAINING and random.random() < self.epsilon:
            return random.choice(acts)
        return max(acts, key=lambda a: self._q_value(wrld, a))

    def _q_value(self, wrld, action):
        return self._q_from_features(self._extract_features(wrld, action))

    def _q_from_features(self, feats):
        return sum(self.weights[f]*v for f,v in feats.items())

    def _max_q(self, wrld):
        acts = self._actions(wrld)
        return max((self._q_value(wrld,a) for a in acts), default=0.0)

    # ---------------- Features & Reward ---------------- #
    def _extract_features(self, wrld, action):
        atype, ax, ay = action
        me = wrld.me(self)
        feats = defaultdict(float)
        tx, ty = me.x+ax, me.y+ay
        feats["bias"] = 1.0
        d = self._dist_exit[tx][ty]
        feats["inv_exit_dist"] = 1.0/(1.0 + (d if d < INF else 9999))
        feats["blocked"] = 1.0 if not path_exists_to_exit(wrld, me.x, me.y) else 0.0
        feats["is_bomb"] = 1.0 if atype == "bomb" else 0.0
        feats["hazard"] = 1.0 if immediate_hazard(tx, ty, wrld) else 0.0
        return feats

    def _compute_reward(self, prev_wrld, curr_wrld, prev_action):
        me_prev = prev_wrld.me(self)
        me_curr = curr_wrld.me(self)
        if me_prev and me_curr is None:
            return R_DEATH
        if me_curr and curr_wrld.exit_at(me_curr.x, me_curr.y):
            return R_EXIT

        reward = R_STEP
        atype, _, _ = prev_action


        # --- Reachability improvement reward ---
        if me_prev and me_curr:
            before = reachable_cells(me_prev.x, me_prev.y, prev_wrld)
            after  = reachable_cells(me_curr.x, me_curr.y, curr_wrld)
            delta  = after - before
            if delta > 0:
                reward += R_BOMB_CLEAR_PATH * delta  # scaled by gain in mobility

        if atype == "bomb":
            # Reward wall destruction
            wb = sum(prev_wrld.wall_at(x,y) for x in range(prev_wrld.width()) for y in range(prev_wrld.height()))
            wa = sum(curr_wrld.wall_at(x,y) for x in range(curr_wrld.width()) for y in range(curr_wrld.height()))
            if wa < wb: reward += R_BOMB_WALL * (wb - wa)
            # Reward clearing path
            if not path_exists_to_exit(prev_wrld, me_prev.x, me_prev.y) and path_exists_to_exit(curr_wrld, me_prev.x, me_prev.y):
                reward += R_CLEAR_PATH
            # Reward monster kills
            mb = sum(bool(prev_wrld.monsters_at(x,y)) for x in range(prev_wrld.width()) for y in range(prev_wrld.height()))
            ma = sum(bool(curr_wrld.monsters_at(x,y)) for x in range(curr_wrld.width()) for y in range(curr_wrld.height()))
            if ma < mb: reward += R_MONSTER_KILL*(mb-ma)

        return reward

    def _should_offer_bomb(self, wrld):
        me = wrld.me(self)
        if immediate_hazard(me.x, me.y, wrld): return False
        if not path_exists_to_exit(wrld, me.x, me.y): return True
        return False

    # ---------------- Persistence ---------------- #
    def save_weights(self):
        with open(WEIGHT_FILE, "w") as f:
            json.dump(self.weights, f)
        print(f"[QL] Saved weights to {WEIGHT_FILE}")

    def load_weights(self):
        with open(WEIGHT_FILE) as f:
            data = json.load(f)
        self.weights = defaultdict(float, {k: float(v) for k,v in data.items()})

