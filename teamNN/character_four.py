# This is necessary to find the main code
import sys
sys.path.insert(0, '../Bomberman')

# Import necessary stuff
from entity import CharacterEntity # type: ignore
from colorama import Fore, Back

class CharacterFour(CharacterEntity):

    def do(self, wrld):
        # Commands
        dx, dy = 0, 0
        bomb = False # don't need to worry about that rn

        # find the exit coordinate
        exit_pos = self.find_exit(wrld)

        # use functions to calc next best move using minimax        
        best_move = self.minimax_decision(wrld, exit_pos, depth=3)
        dx, dy = best_move

        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()
        pass

    def find_exit(self,wrld):
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x,y):
                    return (x,y)
        return None
    
    def in_bounds(self, wrld, x,y):
        if 0 <= x < wrld.width() & 0 <= y < wrld.height(): 
            return True
        else: 
            return False
    
    def move_valid(self, wrld, new_x, new_y):
        if self.in_bounds(wrld,new_x, new_y):
            if wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y):
                return True
            else: 
                return False
    
    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    
    def eval_position(self,wrld, pos, target):
        '''checks all potential positions and returns the potential score of that position'''
        # define "score" (init at 0)
        score = 0

        # penalty weights
        exit_dist_weight = 10
        target_weight = 1000
        monster_weight = 500
        danger_lvl_weight = 25

        # distance to exit (rescore based on dist, closer = better)
        exit_distance = self.manhattan_dist(pos, target)
        score -= exit_distance * exit_dist_weight

        # exit bonus
        if pos == target: 
            score += target_weight

        # penalties (monster is the only one for now)
            if wrld.monsters_at(pos): score -= monster_weight
            # maybe have bombs in the future?

        # check near by the player (& keep counter?)
        danger_lvl = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                potential_x = dx + pos[0]
                potential_y = dy + pos[1]
                if self.in_bounds(wrld, potential_x, potential_y) and wrld.monsters_at(potential_x, potential_y):
                    danger_lvl += 1

        # calculate and return score based on "danger level" or something like that...
        score -= danger_lvl * danger_lvl_weight
        return score
        
    def minimax_desc(self, wrld, target, depth):

        # potential moves: list[tuple(int)]
        moveset = [(-1,-1), (0,-1),(1,-1), # if there is a bug it probably came from a typo here
                   (-1, 0), (0, 0),(1, 0),
                   (-1, 1), (0, 1),(1, 1)]
        
        # best move and best score
        best_move = (0,0)
        best_score = -10000  # need super small value (look into syntax... )
    
        for dx, dy in moveset: # check all surrounding moves
            new_x = dx + self.x
            new_y = dy + self.y

            # get score for this move if it is valid (use function above)
            if self.move_valid(wrld, new_x, new_y):
                score = self.minimax_function(wrld, target, depth - 1,False, (new_x, new_y) ) #TODO: write minimax function (it goes here!) 

                # statement asking if caluclated move is better than the current best
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)

        return best_move
    
    def minimax_function(self, wrld, target, depth, is_player_turn, player_coords):
        # weights / constants
        monster_weight = 100
        max_score_var = 10000
        min_score_var = -10000

        # case 1: player reached exit
        if player_coords == target or depth == 0:
            return self.eval_position(wrld,player_coords, target)
        
        # case 2: player's turn (maximizing)
        if is_player_turn:
            max_score = max_score_var
            moveset = [(-1,-1), (0,-1),(1,-1), # if there is a bug it probably came from a typo here
                       (-1, 0), (0, 0),(1, 0),
                       (-1, 1), (0, 1),(1, 1)]

            for dx, dy in moveset:
                new_x = player_coords[0] + dx
                new_y = player_coords[1] + dy

                if self.move_valid(wrld, new_x, new_y): 
                    new_coords = (new_x, new_y)
                    score = self.minimax_function(wrld, target, depth - 1, False,new_coords) # recursive call (flipped boolean to false)
                    max_score = max(max_score, score)

            return max_score
        
        else:  # case 3: monster's turn (minimizing)
            min_score = min_score_var
            base_score = self.eval_position(wrld, player_coords, target)

            # case A: no change in threat, case B: monster proximity score
            score_a = self.minimax_function(wrld, target, depth-1, True, player_coords) 
            score_b = base_score - monster_weight 

            min_score = min(score_a, score_b)

            return min_score
