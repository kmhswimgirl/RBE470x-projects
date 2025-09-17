# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import numpy as np
import heapq

class AStarCharacter(CharacterEntity):


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
    def plan_path(self, map_array, start, goal) -> list:
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

        # Movement: 4 directions (up, down, left, right)
        moves = [(0,1), (0,-1), (1,0), (-1,0)]

        # Priority queue (f-score, g-score, node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))  # (f, g, (x,y))

        came_from = {}   # for path reconstruction
        g_score = {start: 0}

        def heuristic(a, b):
            # Manhattan distance
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

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
        # look at the current world and determine a path using A* from the current position to the goal
        map_array, start, goal = self.create_map(wrld)

        # Run the A* algorithm here to find a path
        path = self.plan_path(map_array, start, goal)

        # Convert the list of coordinates into movement commands (dx, dy)
        if path:
            moves = self.convert_path_to_moves(path)
            print("Moves:", moves)  # Debugging output

            # Execute the first move in the list
            if moves:
                dx, dy = moves[0]
                self.move(dx, dy)
        else:
            print("No path found!")
        
