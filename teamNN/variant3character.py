# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import numpy as np
import heapq

class Variant3Character(CharacterEntity):
    def determine_state(self, start, danger_map):
        """Determine if the character should be in 'evade' or 'search' state based on danger map."""
        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                nx, ny = start[0] + ddx, start[1] + ddy
                if 0 <= nx < danger_map.shape[0] and 0 <= ny < danger_map.shape[1]:
                    if danger_map[nx][ny] >= 1:
                        return 'evade'
        return 'search'

    def select_safest_move(self, start, map_array, danger_map):
        """Select the safest adjacent move (8 directions). Returns (dx, dy)."""
        best_score = float('inf')
        best_move = (0,0)
        for ddx, ddy in [
            (-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(-1,1),(1,-1),(1,1)
        ]:
            nx, ny = start[0] + ddx, start[1] + ddy
            if 0 <= nx < map_array.shape[0] and 0 <= ny < map_array.shape[1]:
                if map_array[nx][ny] not in (1,2,4):
                    score = danger_map[nx][ny]
                    if score < best_score:
                        best_score = score
                        best_move = (ddx, ddy)
        return best_move

    def safe_plan_path(self, map_array, start, goal, danger_map):
        """A* pathfinding that avoids danger cells."""
        rows, cols = map_array.shape
        moves = [
            (0,1), (0,-1), (1,0), (-1,0),
            (1,1), (1,-1), (-1,1), (-1,-1)
        ]
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}
        def heuristic(a, b):
            dist = abs(a[0]-b[0]) + abs(a[1]-b[1])
            # Add penalty for danger
            if danger_map[a[0]][a[1]] == 1:
                dist += 10000
            elif danger_map[a[0]][a[1]] == 2:
                dist += 5000
            return dist
        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            for dx, dy in moves:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                if map_array[neighbor] in (1,2,4):
                    continue
                if danger_map[neighbor[0]][neighbor[1]] == 1:
                    continue
                tentative_g = g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
        return None
    def get_danger_map(self, wrld):
        """
        Returns a danger map (numpy array) where:
        1 = danger (monster current position, predicted path, or possible turn)
        2 = high risk (adjacent cells at intersections)
        0 = safe
        """
        map_array = np.zeros((wrld.width(), wrld.height()), dtype=int)
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                # Mark monster's current position
                if wrld.monsters_at(x, y):
                    map_array[x][y] = 1
                    # Predict monster's path (straight-line until obstacle)
                    for monster in wrld.monsters_at(x, y):
                        dx, dy = monster.dx, monster.dy
                        nx, ny = x, y
                        while True:
                            nx += dx
                            ny += dy
                            if not (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
                                break
                            if wrld.wall_at(nx, ny) or wrld.bomb_at(nx, ny):
                                break
                            map_array[nx][ny] = 1
                        # Mark adjacent cells at intersections (possible turns)
                        for ddx in [-1, 0, 1]:
                            for ddy in [-1, 0, 1]:
                                tx, ty = x + ddx, y + ddy
                                if (ddx != 0 or ddy != 0) and 0 <= tx < wrld.width() and 0 <= ty < wrld.height():
                                    if not wrld.wall_at(tx, ty) and not wrld.bomb_at(tx, ty):
                                        map_array[tx][ty] = max(map_array[tx][ty], 2)
        return map_array
    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.state = 'search'  # Initial state

    def create_map(self, world) -> (np.ndarray, tuple, tuple):
        """
        Create a 2D numpy array representation of the game world.
        :param world: The game world object.

        :returns:
            map_array (np.ndarray): 2D array representing the game world.
            playerPosition (tuple): Coordinates of the player.
            goal (tuple): Coordinates of the goal (exit).
        """
        
        # Create a 2D numpy array to represent the map
        map_array = np.zeros((world.width(), world.height()))

        # Define start and goal positions
        playerPosition = None  # To be determined based on the character's position
        goal = None  # To be determined based on the world state

        # Fill the array based on the world state
        for x in range(world.width()):
            for y in range(world.height()):
                # Check for walls, bombs, exits, monsters, and players
                if world.wall_at(x, y):
                    map_array[x][y] = 1  # Wall
                elif world.bomb_at(x, y):
                    map_array[x][y] = 2  # Bomb
                elif world.exit_at(x, y):
                    map_array[x][y] = 3  # Exit
                    goal = (x, y)  # Set goal position
                elif world.monsters_at(x, y):
                    map_array[x][y] = 4  # Monster
                elif world.characters_at(x, y):
                    map_array[x][y] = 5  # Player
                    playerPosition = (x, y) # Set player position
                else:
                    map_array[x][y] = 0  # Empty space
        return map_array, playerPosition, goal

    # Plan a path using A* algorithm
    def plan_path(self, map_array, start, goal) -> list | None:
        '''
        Plan a path using the A* algorithm from start to goal on the given map_array.
        Should return a list of coordinates representing the path.
        
        :param map_array: 2D numpy array representing the game world.
        :param start: Tuple (x, y) representing the starting position.
        :param goal: Tuple (x, y) representing the goal position.

        :returns:
            path (list): List of tuples representing the path from start to goal.
        '''

        path = []

        rows, cols = map_array.shape

        # Movement: 8 directions (including diagonals)
        moves = [
            (0,1), (0,-1), (1,0), (-1,0),
            (1,1), (1,-1), (-1,1), (-1,-1)
        ]

        # Priority queue (f-score, g-score, node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))  # (f, g, (x,y))

        came_from = {}   # for path reconstruction
        g_score = {start: 0}


        def heuristic(a, b):
            # Find the Manhattan distance to the goal
            dist = abs(a[0]-b[0]) + abs(a[1]-b[1])

            # Find the minimum distance to any monster
            min_dist_to_monster = float('inf')
            for x in range(rows):
                for y in range(cols):
                    if map_array[x][y] == 4: # monster or inflated monster cell
                        d = abs(a[0]-x) + abs(a[1]-y)
                        if d < min_dist_to_monster:
                            min_dist_to_monster = d

            # Scale penalty/reward based on distance to monster
            if min_dist_to_monster <= 5:
                dist += 100 / (min_dist_to_monster + 1)  # Strong penalty for being very close
            else:
                dist -= 100 * min_dist_to_monster  # Reward for being far

            return dist

        while open_set:
            _, g, current = heapq.heappop(open_set)

            if current == goal:
                # reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # reverse

            for dx, dy in moves:
                neighbor = (current[0] + dx, current[1] + dy)

                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue  # skip out of bounds
                if map_array[neighbor] in (1,2,4): 
                    continue  # skip obstacles

                tentative_g = g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current

        return None  # no path found

    def convert_path_to_moves(self, path):
        '''
        Convert a list of coordinates into movement commands (dx, dy).

        :param path: List of tuples representing the path.

        :returns:
            moves (list): List of movement commands as (dx, dy).
        '''
        moves = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            moves.append((dx, dy))
        return moves

    def do(self, wrld):
        map_array, start, goal = self.create_map(wrld)
        danger_map = self.get_danger_map(wrld)
        self.state = self.determine_state(start, danger_map)

        if self.state == 'search':
            path = self.safe_plan_path(map_array, start, goal, danger_map)
            if path:
                moves = self.convert_path_to_moves(path)
                if moves:
                    dx, dy = moves[0]
                    self.move(dx, dy)
            else:
                print("No path found!")
        elif self.state == 'evade':
            dx, dy = self.select_safest_move(start, map_array, danger_map)
            self.move(dx, dy)