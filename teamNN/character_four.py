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
        self.cached_world = wrld # ref for monster calcs
        
        open_set = [(0, 0, start)]
        heapq.heapify(open_set)
        
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(wrld, start, goal)}
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set) # may look unused, but it unwraps the tuple
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            if current == goal:
                self.cached_world = None
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
                
                tentative_g = g_score[current] + self.cost_with_monsters(wrld, current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(wrld, neighbor, goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        self.cached_world = None
        return None

    def heuristic(self, wrld, pos, goal):
        '''2 block safety radius from the monster'''

        # initial penalties
        base_distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        monster_penalty = 0
        future_penalty = 0
        
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.monsters_at(x, y):
                    monster_pos = (x, y) # current monster position
                    current_player_pos = (self.x, self.y) # current player position
                    
                    distance_to_monster = abs(pos[0] - x) + abs(pos[1] - y) # calc normal dist to monster
                    
                    # scale penalties based on distance, since i want a safe zone radius r = 2, exponentially tax the distances =< 2
                    map_penalty = {0: 60000, 1: 20000, 2: 5000, 3: 100, 4: 20, 5: 5}
                    monster_penalty += map_penalty.get(distance_to_monster, 0)    
                    
                    predicted_monster_moves = self.predict_monster_movement(
                        monster_pos, current_player_pos, wrld
                    )
                    
                    # potential monster positions
                    for future_monster_pos in predicted_monster_moves:
                        future_distance = abs(pos[0] - future_monster_pos[0]) + abs(pos[1] - future_monster_pos[1])
                        if future_distance == 0:
                            future_penalty += 40000  # same square
                        elif future_distance == 1:
                            future_penalty += 15000  # adjacent
                        elif future_distance == 2:
                            future_penalty += 2550   # under rad = 2
                        elif future_distance == 3:
                            future_penalty += 200    # not that bad of a threat
                    
                    if self.is_in_corridor(wrld, pos) and distance_to_monster <= 6:
                        monster_penalty += 500  # small corridors = not good
                    
                    escape_routes = self.count_escape_routes(wrld, pos)
                    if escape_routes <= 2 and distance_to_monster <= 5: # see how many viable escape paths there could be
                        monster_penalty += 800  
                    
                    if self.trap_scenario(wrld, pos, monster_pos):
                        monster_penalty += 1000
        
        # progress points to hopefully prevent oscillating
        progress_bonus = 0
        current_distance_to_goal = abs(current_player_pos[0] - goal[0]) + abs(current_player_pos[1] - goal[1])
        new_distance_to_goal = base_distance
        
        if new_distance_to_goal < current_distance_to_goal:
            progress_bonus = -10 

        # wall avoidance        
        obstacle_penalty = 0
        if self.count_escape_routes(wrld, pos) <= 1:
            obstacle_penalty += 1000  # dead ends are very not good

        esc_score = self.count_escape_routes(wrld, pos)
        obstacle_penalty += (8 - esc_score) * 20  # low mobility also bad
        
        total_heuristic = (base_distance + monster_penalty + future_penalty + progress_bonus + obstacle_penalty)
        
        return max(0, total_heuristic)

    def cost_with_monsters(self, wrld, from_pos, to_pos):
        # base movement cost
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        base_cost = 1.4 if (dx == 1 and dy == 1) else 1.0
        
        # monster danger cost
        monster_cost = 0
        
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.monsters_at(x, y):
                    distance_to_monster = abs(to_pos[0] - x) + abs(to_pos[1] - y)
                    
                    if distance_to_monster == 0:
                        monster_cost += 25000  # die
                    elif distance_to_monster == 1:
                        monster_cost += 10000  # inside circle
                    elif distance_to_monster == 2:
                        monster_cost += 2500   # right on the line
                    elif distance_to_monster == 3:
                        monster_cost += 50     # not good not bad
                    elif distance_to_monster == 4:
                        monster_cost += 10     # fine ig
        return base_cost + monster_cost

    def predict_monster_movement(self, monster_pos, player_pos, wrld):
        possible_moves = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            new_x = monster_pos[0] + dx
            new_y = monster_pos[1] + dy
            
            if (self.in_bounds(wrld, new_x, new_y) and 
                (wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y))):
                
                # monster should prefers moves tha get closer to player
                old_distance = abs(monster_pos[0] - player_pos[0]) + abs(monster_pos[1] - player_pos[1])
                new_distance = abs(new_x - player_pos[0]) + abs(new_y - player_pos[1])
                
                if new_distance <= old_distance:  # monster approaches or remains the same
                    possible_moves.append((new_x, new_y))
        
        return possible_moves if possible_moves else [monster_pos]  # stay in the same spot if there are no good moves left
    
    def is_in_corridor(self, wrld, pos): # lightly trapped in a corridor
        free_directions = 0 # directions that are free to move in 
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # n e s w
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if (self.in_bounds(wrld, new_x, new_y) and 
                (wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y))):
                free_directions += 1
        
        return free_directions <= 2  # corridor by def has 2 or less exits (hallway)
    
    def count_escape_routes(self, wrld, pos): # how many ways could you escape the monster (i.e. pinned against a wall = 2)
        escape_count = 0
        
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if (self.in_bounds(wrld, new_x, new_y) and 
                (wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y))):
                escape_count += 1
        return escape_count

    def is_valid_position(self, wrld, x, y): # another is valid as there were occasionally issues for some reason beyond me
        if not self.in_bounds(wrld, x, y):
            return False
        
        if not (wrld.empty_at(x, y) or wrld.exit_at(x, y)):
            return False
        
        for mx in range(wrld.width()):
            for my in range(wrld.height()):
                if wrld.monsters_at(mx, my):
                    distance_to_monster = abs(x - mx) + abs(y - my)
                    if distance_to_monster < 3: # inside radius
                        return False  
        return True

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
    
    def in_bounds(self, wrld, x,y): # could prob get rid of this.. only like one line
        return 0 <= x < wrld.width() and 0 <= y < wrld.height()
    
    def trap_scenario(self, wrld, pos, monster_pos):
        '''scenario B: trap = less than 2 safe exits'''
        esc_pos = []
        
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            esc_x = pos[0] + dx
            esc_y = pos[1] + dy
            
            if (0 <= esc_x < wrld.width() and 0 <= esc_y < wrld.height()) and (wrld.empty_at(esc_x, esc_y) or wrld.exit_at(esc_x, esc_y)):
                # does the escape position keep the safety radius
                esc_distance = abs(esc_x - monster_pos[0]) + abs(esc_y - monster_pos[1])
                if esc_distance >= 3:  # determined to be safe esc route
                    esc_pos.append((esc_x, esc_y))
        return len(esc_pos) < 2
