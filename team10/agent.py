# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math
import heapq
from enum import Enum
import numpy as np
import random
import os

# Set to True to enable training (weights are updated and saved).
# Set to False to run in inference mode (weights are loaded but not changed).
TRAINING = True

class World_State(Enum):
    WORLD_SAFE = 0      # not used
    WORLD_HAS_BOMB = 1


class TestCharacter(CharacterEntity):
    weights_file = 'weights'

    def __init__(self, name, avatar, x, y, variant_num):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.variant_num = variant_num
        # keep track of bomb explosion time
        self.my_bomb_timer = math.inf

    def do(self, world):
        """Decide and perform the next action for this character.

        This is the main per-timestep method called by the game engine. It
        inspects the world state (bombs, monsters, explosions), updates the
        character's internal bomb timer, and either navigates toward the exit
        (using A*) or runs the approximate Q-learning routine to pick an action
        when the world is unsafe.

        Parameters
        - wrld: World
            The current game world object exposing methods like
            `wall_at`, `explosion_at`, `empty_at`, `bomb_time`, and `exitcell`.

        Side effects
        - Moves the character via `self.move`.
        - May place a bomb via `self.place_bomb`.
        - May read and write weight files using `read_weights`/`write_weights`.
        - Updates `self.my_bomb_timer` when a bomb is placed.
        """
        # always WORLD_HAS_BOMB
        if len(self.get_bomb_location(world)) == 0 and len(self.get_monster_location(world)) == 0 and len(self.get_explosion_location(world)) == 0:
            world_state = World_State.WORLD_SAFE
        else:
            world_state = World_State.WORLD_HAS_BOMB

        # reduce timer
        if self.my_bomb_timer == 0:
            self.my_bomb_timer = math.inf
        # math.inf-1 == math.inf, but math.inf-1 is not math.inf. Use == for comparison
        # so nothing will change if timer = inf (no bomb)
        self.my_bomb_timer -= 1

        # check if close to exit
        near_exit = max(abs(world.exitcell[0] - self.x), abs(world.exitcell[1] - self.y)) <= 1
        if near_exit:
            self.move(world.exitcell[0] - self.x, world.exitcell[1] - self.y)
            return

        # while the character is not at exit yet
        match world_state:
            # perform A* if world is safe
            case World_State.WORLD_SAFE:
                # get path to exit
                path = self.astar(self, world)
                if not path:
                    return
                next_cell = path[0]
                # find the interaction point between the path and wall
                if not world.wall_at(next_cell[0], next_cell[1]):
                    self.move(next_cell[0] - self.x, next_cell[1] - self.y)
                elif world.explosion_at(next_cell[0], next_cell[1]):
                    self.move(0, 0)
                else:
                    # place a bomb if the next move is a wall
                    self.place_bomb()
                    self.my_bomb_timer = world.bomb_time + 1
                    world_state = World_State.WORLD_HAS_BOMB

            # Q learning if world is not safe
            case World_State.WORLD_HAS_BOMB:
                weight = self.read_weights()
                # approximate_Q will perform action selection; pass TRAINING
                new_weight = self.approximate_Q(weight, world, train=TRAINING)
                # only save updated weights when training is enabled
                if TRAINING:
                    self.write_weights(new_weight)


    def read_weights(self):
        """Read and return the weight vector for this character variant.

        Behavior:
        - Attempts to read weights from a file named `weights<variant_num>`.
        - If that file does not exist, falls back to
          `weights<variant_num>.txt`.

        Returns
        - list[float]: list of weights read from the file.

        Notes
        - If the file contains non-numeric lines a ValueError will be caught
          and an error message will be printed (the exception is not re-raised).
        """
        # Use explicit .txt suffix for weight files: weights<variant>.txt
        self.weights_file = 'weights' + str(self.variant_num) + '.txt'
        if os.path.exists(self.weights_file):
            try:
                weights = [float(line.rstrip('\n')) for line in open(self.weights_file, 'r')]
            except ValueError:
                print('ERROR (ValueError): unexpected newline character encountered')
        else:
            weights = [float(line.rstrip('\n')) for line in open('weights' + str(self.variant_num) + '.txt', 'r')]
        return weights

    def write_weights(self, weight):
        """Write the weight vector to the configured weights file.

        Parameters
        - weight: iterable of numbers
            The weights to write to `self.weights_file`, one per line.

        Side effects
        - Overwrites the file referenced by `self.weights_file`.

        Errors
        - If `weight` is None or not iterable, a TypeError is caught and an
          error message is printed.
        """
        # Ensure the weights file name follows the pattern weights<variant>.txt
        with open(self.weights_file, 'w') as f:
            try:
                f.writelines(['%s\n' % w for w in weight])
            except TypeError:
                print('ERROR (TypeError): weights is empty (is None)')

    # perform approximate q algorithm
    def approximate_Q(self, weight, world, train: bool = True):
        """Perform one approximate Q-learning update and choose an action.

        This routine evaluates all immediate successor states (including a
        hypothetical bomb placement) using the provided linear weight vector
        and a feature extractor. It selects the action with the highest
        estimated Q-value, executes that action in the real world (or places
        a bomb), and then performs a gradient update on the weight vector
        using the observed reward and the estimate of max_a' Q(s', a').

        Parameters
        - weight: list[float]
            Current weight vector for the linear Q-function approximation. The
            vector is updated in-place and also returned.
        - wrld: World
            The current world. This method clones the world (via
            `wrld.from_world`) to simulate possible actions.

        Returns
        - list[float]: the updated weight vector.

        Notes and assumptions
        - The method assumes `get_feature_vector` returns a feature vector of
          the same length as `weight`.
        - Rewards are taken from `new_wrld.scores[char.name]` and offset by a
          constant to keep values positive.
        - The method uses small learning rate `alpha=0.01` and discount
          factor `gamma=0.9`. These are hard-coded hyperparameters.
        """
        # hyperparams
        gamma = 0.9  # Discount factor
        alpha = 0.01  # Learning rate
        # alpha = 0.000001  # Learning rate
        # features:
        #   distance to bomb, time left for explosion, in corner, distance to exit, distance to closest wall, dist to monst, in explosion range
        # possible actions:
        #   move up, move down, move left, move right, place bomb
        possible_actions = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        # dictionaries for rewards and Q-values for next possible actions
        rewards = {}
        q_values = {}
        bomb_timers = {}  # location: bomb remaining time

        # evaluate each possible immediate action by simulating it
        for action in possible_actions:
            next_location = (self.x + action[0], self.y + action[1])
            next_x, next_y = next_location
            # create a simulated copy of the world and character to evaluate the action
            sim_world = world.from_world(world)
            bomb_timers[next_location] = math.inf

            if self.check_inbound(next_x, next_y, sim_world) and not self.in_explosion_range(next_x, next_y, world):
                sim_char = list(sim_world.characters.values())[0][0]

                if sim_world.empty_at(next_x, next_y):
                    sim_char.move(action[0], action[1])
                else:
                    # not valid move, try placing a bomb instead (only if none exist)
                    if len(self.get_bomb_location(world)) == 0 and sim_world.wall_at(next_x, next_y) and len(self.get_explosion_location(world)) == 0:
                        sim_char.place_bomb()
                        bomb_timers[next_location] = sim_world.bomb_time + 1
                    else:
                        continue

                # reward for this simulated step (offset to keep positive)
                reward = sim_world.scores[sim_char.name] + 5000

                rewards[next_location] = reward
                next_features = self.get_feature_vector(next_x, next_y, sim_world)
                q_value = np.dot(weight, next_features)
                q_values[next_location] = q_value

        # collect s' and reward
        # select the key with the maximum Q-value
        # (if q_values is empty this will raise; that matches previous behavior)
        chosen_move = max(q_values.items(), key=lambda kv: kv[1])[0]
        chosen_reward = rewards[chosen_move]
        chosen_q = q_values[chosen_move]
        # copy bomb state if the simulated best action placed a bomb
        if bomb_timers[chosen_move] != math.inf:
            self.place_bomb()
            self.my_bomb_timer = bomb_timers[chosen_move]
        else:
            # execute the chosen movement
            self.move(chosen_move[0] - self.x, chosen_move[1] - self.y)

        current_features = self.get_feature_vector(self.x, self.y, world)

        next_q_values = {}
        # find max Q(s', a') over possible next actions from the chosen state
        for action in possible_actions:
            next_location = (chosen_move[0] + action[0], chosen_move[1] + action[1])
            sim_world = world.from_world(world)
            sim_char = next(iter(sim_world.characters.values()))[0]
            sim_char.move(action[0], action[1])
            next_x = next_location[0]
            next_y = next_location[1]
            if self.check_inbound(next_x, next_y, sim_world) and sim_world.empty_at(next_x, next_y) and not self.in_explosion_range(next_x, next_y, sim_world):
                next_features = self.get_feature_vector(next_x, next_y, sim_world)
                q_val = np.dot(weight, next_features)
                next_q_values[next_location] = q_val
        if next_q_values:
            next_q = max(next_q_values.values())
        else:
            next_q = 0

        # set current reward and Q (estimates computed earlier)
        current_reward = chosen_reward
        current_q = chosen_q

        # calculate delta (TD error)
        delta = (current_reward + gamma * next_q) - current_q

        # recalculate weight only if in training mode
        if train:
            for i in range(len(weight)):
                f = current_features[i]
                weight[i] = weight[i] + alpha * delta * f

        return weight


    def get_feature_vector(self, x, y, wrld):
        """Return a feature vector for the state when the character is at (x,y).

        The feature vector is used by the linear Q-function approximation. Each
        element is a numeric feature which should align with the weight
        vector length expected by `approximate_Q`.

        Parameters
        - x, y: int
            Coordinates of the candidate state.
        - wrld: World
            The world used to compute features (exit location, monster
            positions, etc.).

        Returns
        - list[float]: feature vector, currently [distance-to-exit,
          distance-to-closest-monster].
        """
        exit_dist = self.dist_to_exit(x, y, wrld)
        monst_dist = self.dist_to_closest_monst(x, y, wrld)
        feature_vector = [exit_dist, monst_dist]
        return feature_vector


    # estimate distance to the exit cell
    def dist_to_exit(self, x, y, wrld):
        """Estimate and return a scaled inverse distance to the exit.

        The returned value is in (0,1], higher values indicate closer to the
        exit. The particular formula applies a stronger weight to vertical
        distance (multiplied by 10) which is a design choice in this agent.

        Parameters
        - x, y: int
            Coordinates to measure from.
        - wrld: World
            World object containing `exitcell`.

        Returns
        - float: scaled inverse Euclidean distance to exit in (0,1].
        """
        return 1 / (1+math.sqrt((wrld.exitcell[0]-x)**2 + 10*(wrld.exitcell[1]-y)**2))
        # return 1 / (1+math.sqrt((wrld.exitcell[0]-x)**2 + (wrld.exitcell[1]-y)**2))

    # get distance to the closest monster
    def dist_to_closest_monst(self, x, y, wrld):
        """Return a bounded feature reflecting proximity to the closest monster.

        The feature is a smoothed, sigmoid-like transform of Euclidean
        distance such that the value is near 1 when monsters are very close
        (discourage moves that approach monsters) and near 0 when monsters are
        far away.

        Parameters
        - x, y: int
            Coordinates to evaluate.
        - wrld: World
            World object used to find monsters via `get_monster_location`.

        Returns
        - float: value in (0,1] inversely related to distance to nearest
          monster.
        """
        monst_loc_list = self.get_monster_location(wrld)
        dist_to_closest_monst = math.inf
        for monst_loc in monst_loc_list:
            dist_to_monst = math.sqrt((monst_loc[0]-x)**2 + (monst_loc[1]-y)**2)
            dist_to_closest_monst = min(dist_to_closest_monst, dist_to_monst)

        d = dist_to_closest_monst
        # engineer a sigmoid like function to avoid close contact to monster, 
        # while being less sensitive to far away monster
        return 1 / (1 + (d/1.5)**4)


    
    # check whether character is in the explosion range
    def in_explosion_range(self, x, y, wrld):
        """Check whether position (x,y) would be hit by an explosion.

        Logic:
        - If there are no bombs reported in the world, returns whether an
          explosion is currently active at (x,y) using `wrld.explosion_at`.
        - If a bomb exists, when this agent's `my_bomb_timer` indicates the
          bomb is about to explode (<= 1) it uses the bomb location and the
          world's `expl_range` to determine if (x,y) lies in the cross-shaped
          blast zone (same row or column within range). Otherwise it falls
          back to checking `wrld.explosion_at`.

        Parameters
        - x, y: int
            Coordinates to test.
        - wrld: World
            World object providing bomb and explosion information.

        Returns
        - bool: True if (x,y) is considered in explosion range or currently
          exploding; False otherwise.
        """
        bombLoc = self.get_bomb_location(wrld)
        if len(bombLoc) == 0:
            return bool(wrld.explosion_at(x,y))
        bombLoc = self.get_bomb_location(wrld)[0]

        # run if about to blow up
        if self.my_bomb_timer <= 1:
            if x < (bombLoc[0] + wrld.expl_range + 1) and x > (bombLoc[0] - wrld.expl_range - 1) and y == bombLoc[1]:
                return 1
            if y < (bombLoc[1] + wrld.expl_range + 1) and y > (bombLoc[1] - wrld.expl_range - 1) and x == bombLoc[0]:
                return 1
            return 0

        return bool(wrld.explosion_at(x,y))




    # get location of bomb in the world
    def get_bomb_location(self, wrld):
        """Return a list of all bomb coordinates currently present in `wrld`.

        Parameters
        - wrld: World
            World object with `width`, `height` and `bomb_at` query methods.

        Returns
        - list[tuple[int,int]]: coordinates (x,y) of bombs.
        """
        bombs = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.bomb_at(i, j):
                    bombs.append((i, j))
        return bombs

    # get location of monsters in the world
    def get_monster_location(self, wrld):
        """Return a list of coordinates where monsters are present in `wrld`.

        Parameters
        - wrld: World
            World object with `width`, `height` and `monsters_at` queries.

        Returns
        - list[tuple[int,int]]: coordinates (x,y) of monsters.
        """
        monsters = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.monsters_at(i, j):
                    monsters.append((i, j))
        return monsters

    def get_explosion_location(self, wrld):
        """Return a list of coordinates currently affected by an explosion.

        Parameters
        - wrld: World
            World object with `width`, `height` and `explosion_at` queries.

        Returns
        - list[tuple[int,int]]: coordinates (x,y) where explosions are active.
        """
        explo_list = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.explosion_at(i, j):
                    explo_list.append((i, j))
        return explo_list

    # check whether the next movement is still inbound
    def check_inbound(self, x, y, wrld):
        """Return True if coordinates (x,y) are within the world bounds.

        Parameters
        - x, y: int
            Coordinates to test.
        - wrld: World
            World object providing `width` and `height`.

        Returns
        - bool: True when 0 <= x < width and 0 <= y < height; False otherwise.
        """
        if x >= 0 and x < wrld.width() and y >=0 and y < wrld.height():
            return True
        return False



    # calculate A* between character and exit
    def astar(self, char, wrld):
        """Compute a path from the character to the world's exit using A*.

        The search considers 8-connected neighbors and treats walls as higher
        cost (multiplied by 10). The heuristic is straight-line (Euclidean)
        distance to the exit. If a path is found, the method returns a list of
        coordinates representing the path (excluding the start position, the
        first element is the immediate next cell to move to).

        Parameters
        - char: CharacterEntity
            The character object providing current coordinates `x` and `y`.
        - wrld: World
            World object providing `exitcell`, grid dimensions and `wall_at`.

        Returns
        - list[tuple[int,int]]: ordered list of coordinates from the next step up
          to and including the exit cell. If already at the exit, returns an
          empty list.
        """
        start = (char.x, char.y)
        end = wrld.exitcell

        frontier = CustomPQ()
        frontier.put(start, 0)        
        came_from = {}
        came_from[start] = None
        cost = {}
        cost[start] = 0
        visited = {}
        visited[start] = True
        
        path = []

        while not frontier.is_empty():
            cur_node = frontier.get()

            if end == cur_node:
                break

            next_node_list = []

            cur_x = cur_node[0]
            cur_y = cur_node[1]
            neighbors = [(cur_x+1, cur_y),
                        (cur_x+1, cur_y+1),
                        (cur_x+1, cur_y-1),
                        (cur_x, cur_y+1),
                        (cur_x, cur_y-1),
                        (cur_x-1, cur_y),
                        (cur_x-1, cur_y+1),
                        (cur_x-1, cur_y-1),]
            for n in neighbors:
                n_x = n[0]
                n_y = n[1]
                if n_x >= 0 and n_y >= 0 and n_x < wrld.width() and n_y < wrld.height():
                    # if wrld.wall_at(n_x, n_y) != True:
                    next_node_list.append(n)
            
            for nn in next_node_list:
                new_cost = cost[cur_node] + wrld.wall_at(nn[0], nn[1])*10
                if (nn not in cost or new_cost < cost[nn]) and nn not in visited:
                    nn_x = nn[0]
                    nn_y = nn[1]
                    cost[nn] = new_cost
                    p = new_cost + math.sqrt((nn_x - end[0])**2 + (nn_y - end[1])**2)
                    frontier.put(nn, p)
                    came_from[nn] = cur_node
                    visited[nn] = True
        
        while cur_node != start:
            path.insert(0, cur_node)
            cur_node = came_from[cur_node]
        
        return path
    
class CustomPQ:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def put(self, element, priority):
        found = False
        for i in range(len(self.items)):
            current_priority, current_element = self.items[i]
            if current_element == element:
                if current_priority > priority:
                    self.items[i] = (priority, element)
                    heapq.heapify(self.items)  # Re-heapify after updating priority
                found = True
                break
        if not found:
            heapq.heappush(self.items, (priority, element))

    def get(self):
        if self.items:
            return heapq.heappop(self.items)[1]
        else:
            raise IndexError("Custom priority queue is empty")