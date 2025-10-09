# qlearn_character.py
import sys
import os
import math
import random
import json
import numpy as np
from colorama import Fore, Back

sys.path.insert(0, '../Bomberman')
from entity import CharacterEntity  # type: ignore
from collections import deque

# ===============================================================
# Configuration
# ===============================================================
TRAINING = True  # <-- set to False when running final evaluations
WEIGHT_FILE = "bomb_q_weights.json"

INF = 1e9
DIRS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
DIRS9 = [(0, 0)] + DIRS8

# ===============================================================
# Q-Learner
# ===============================================================
class ApproxQLearner:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = {
            "bias": 0.0,
            "dist_to_exit": 0.0,
            "near_wall": 0.0,
            "monster_nearby": 0.0,
            "bomb_nearby": 0.0
        }
        self.load_weights()

    # ----------------------------
    # Feature extraction
    # ----------------------------
    def features(self, wrld, me):
        f = {}
        f["bias"] = 1.0

        # distance to exit (Manhattan normalized)
        dist = 9999
        for y in range(wrld.height()):
            for x in range(wrld.width()):
                if wrld.exit_at(x, y):
                    d = abs(me.x - x) + abs(me.y - y)
                    dist = min(dist, d)
        f["dist_to_exit"] = 1.0 / (1.0 + dist)

        # near wall
        f["near_wall"] = 1.0 if any(
            wrld.wall_at(nx, ny)
            for nx in range(me.x - 1, me.x + 2)
            for ny in range(me.y - 1, me.y + 2)
            if (nx, ny) != (me.x, me.y)
            and 0 <= nx < wrld.width()
            and 0 <= ny < wrld.height()
        ) else 0.0

        # nearby monster
        f["monster_nearby"] = 1.0 if any(
            wrld.monsters_at(nx, ny)
            for nx in range(me.x - 2, me.x + 3)
            for ny in range(me.y - 2, me.y + 3)
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height()
        ) else 0.0

        # nearby bomb
        f["bomb_nearby"] = 1.0 if any(
            wrld.bomb_at(nx, ny)
            for nx in range(me.x - 2, me.x + 3)
            for ny in range(me.y - 2, me.y + 3)
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height()
        ) else 0.0

        return f

    # ----------------------------
    # Q-function
    # ----------------------------
    def q_value(self, f):
        return sum(self.weights[k] * f[k] for k in self.weights)

    def choose_action(self, wrld, me):
        """Epsilon-greedy bomb choice (1=bomb, 0=wait)."""
        f = self.features(wrld, me)
        if TRAINING and random.random() < self.epsilon:
            return random.choice([0, 1]), f
        q_drop = self.q_value(f)
        q_no = 0
        return (1, f) if q_drop > q_no else (0, f)

    def update(self, reward, f_prev, f_next, done=False):
        q_prev = self.q_value(f_prev)
        q_next = 0 if done else self.q_value(f_next)
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_prev
        for k in self.weights:
            self.weights[k] += self.alpha * td_error * f_prev[k]

    # ----------------------------
    # Save/Load weights
    # ----------------------------
    def save_weights(self):
        with open(WEIGHT_FILE, "w") as f:
            json.dump(self.weights, f, indent=2)

    def load_weights(self):
        if os.path.exists(WEIGHT_FILE):
            with open(WEIGHT_FILE, "r") as f:
                self.weights = json.load(f)
                print(f"[Q-Learner] Loaded weights from {WEIGHT_FILE}")


# ===============================================================
# Utility functions
# ===============================================================
def in_bounds(x, y, wrld):
    return 0 <= x < wrld.width() and 0 <= y < wrld.height()


# ===============================================================
# Q-Learning Bomberman Character
# ===============================================================
class ApproxQLearningCharacter(CharacterEntity):
    def __init__(self, name="hero", avatar="Q", x=0, y=0):
        super().__init__(name, avatar, x, y)
        self.q_bomb = ApproxQLearner(alpha=0.1, gamma=0.9, epsilon=0.2)
        self._bomb_active = False
        self._bomb_cooldown = 0
        self._BOMB_COOLDOWN_TICKS = 10
        self._last_feats = None
        self._last_action = None

    # ----------------------------------------------------------
    def do(self, wrld):
        me = wrld.me(self)
        if me is None:
            return

        # Decide whether to bomb using Q-learning
        action, feats = self.q_bomb.choose_action(wrld, me)
        self._last_feats = feats
        self._last_action = action

        if action == 1 and not self._bomb_active and self._bomb_cooldown <= 0:
            self.place_bomb()
            self._bomb_active = True
            self._bomb_cooldown = self._BOMB_COOLDOWN_TICKS

        # Move randomly or toward safe open cell (for exploration)
        possible_moves = [(0, 0)] + DIRS8
        random.shuffle(possible_moves)
        for dx, dy in possible_moves:
            nx, ny = me.x + dx, me.y + dy
            if in_bounds(nx, ny, wrld) and not wrld.wall_at(nx, ny):
                self.move(dx, dy)
                break

        # Update cooldown
        if self._bomb_cooldown > 0:
            self._bomb_cooldown -= 1

        # Reset active bomb flag
        if self._bomb_active:
            any_bombs = any(
                wrld.bomb_at(x, y)
                for x in range(wrld.width())
                for y in range(wrld.height())
            )
            if not any_bombs:
                self._bomb_active = False

        # ---------------- Q-learning update ----------------
        reward = self.compute_reward(wrld, me)
        if self._last_feats:
            f_next = self.q_bomb.features(wrld, me)
            done = (
                wrld.time <= 0
                or not any(c is self for chars in wrld.characters.values() for c in chars)
            )
            self.q_bomb.update(reward, self._last_feats, f_next, done=done)

        if TRAINING:
            self.q_bomb.save_weights()

    # ----------------------------------------------------------
    def compute_reward(self, wrld, me):
        """Reward shaping for bomb learning, compatible with world end conditions."""
        reward = -1.0  # small time penalty

        # +100 for reaching the exit
        if wrld.exit_at(me.x, me.y):
            reward += 1000.0

        # +50 for hitting monsters with explosion
        for y in range(wrld.height()):
            for x in range(wrld.width()):
                if wrld.explosion_at(x, y) and wrld.monsters_at(x, y):
                    reward += 50.0

        # -200 if this character is missing from the world (i.e., dead)
        still_alive = any(
            c is self for chars in wrld.characters.values() for c in chars
        )
        if not still_alive:
            reward -= 1000.0

        # -50 if time nearly runs out
        if wrld.time <= 2:
            reward -= 50.0

        return reward

