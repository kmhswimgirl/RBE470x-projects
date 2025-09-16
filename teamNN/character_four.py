# This is necessary to find the main code
import sys
sys.path.insert(0, '../Bomberman')

# Import necessary stuff
from entity import CharacterEntity # type: ignore
from colorama import Fore, Back
import heapq

class CharacterFour(CharacterEntity):    
    def do(self, wrld):
        # Commands
        dx, dy = 0, 0
        bomb = False

        # find the exit coordinate
        exit_pos = self.find_exit(wrld)
        if not exit_pos:
            self.move(dx, dy)
            return

        path = self.a_star(wrld, (self.x, self.y), exit_pos)
        
        if path and len(path) > 1:
            next_pos = path[1]  # path[0] is current position
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
        else:
            dx, dy = 0, 0

        self.move(dx, dy) 
        print(f"Current: ({self.x}, {self.y}), Target: {exit_pos}, Moving: ({dx}, {dy})")

        if bomb:
            self.place_bomb()

    def a_star(self, wrld, start, goal):
        
        open_set = [(0, 0, start)]
        heapq.heapify(open_set)
        
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            neighbors = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            ]
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                    
                if not self.is_valid_position(wrld, neighbor[0], neighbor[1]):
                    continue
                
                tentative_g = g_score[current] + self.movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        return None

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def movement_cost(self, from_pos, to_pos):
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        if dx == 1 and dy == 1:
            return 1.4  # diagonals
        else:
            return 1.0  # N, E, S, W

    def is_valid_position(self, wrld, x, y):
        if not self.in_bounds(wrld, x, y):
            return False
        
        if wrld.empty_at(x, y) or wrld.exit_at(x, y):
            return True
            
        return False

    def reconstruct_path(self, came_from, current):
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path

    def find_exit(self,wrld):
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x,y):
                    return (x,y)
        return None
    
    def in_bounds(self, wrld, x,y):
        return 0 <= x < wrld.width() and 0 <= y < wrld.height()
    
    def move_valid(self, wrld, new_x, new_y):
        if self.in_bounds(wrld,new_x, new_y):
            return wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y)
        return False
