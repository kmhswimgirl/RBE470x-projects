# This is necessary to find the main code
import sys
sys.path.insert(0, '../Bomberman')
import math

# Import necessary stuff
from entity import CharacterEntity # type: ignore
from colorama import Fore, Back

class CharacterOne(CharacterEntity):

    def do(self, wrld):
        # Commands
        dx, dy = 0, 0
        bomb = False # don't need to worry about that rn

        # find the exit coordinate
        exit_pos = self.find_exit(wrld)

        # minimax calculates next best move (best_move = )

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
        best_score = "placeholder" # need super small value (look into syntax)
    
        for dx, dy in moveset: # check all surrounding moves
            new_x = dx + self.x
            new_y = dy + self.y

            # get score for this move if it is valid (use function above)
            if self.move_valid(wrld, new_x, new_y):
                score = "placeholder" #TODO: write minimax function (it goes here!) 

                # statement asking if caluclated move is better than the current best
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)

        return best_move
    
    def minimax_function(self, wrld, target, depth, is_player_turn, player_coords):
        # case 1: player reached exit
        if player_coords == target or depth == 0:
            return self.eval_position(wrld,player_coords, target)
        
        # case 2: player's turn (maximizing)
        if is_player_turn:
            max_score = "placeholder"
            moveset = [(-1,-1), (0,-1),(1,-1), # if there is a bug it probably came from a typo here
                       (-1, 0), (0, 0),(1, 0),
                       (-1, 1), (0, 1),(1, 1)]

            for dx, dy in moveset:
                new_x = player_coords[0] + dx
                new_y = player_coords[1] + dy


                if self.move_valid(wrld, new_x, new_y):
                    new_coords = (new_x, new_y)
                    score = self.minimax_function(wrld, target, depth + 1, False,new_coords) # recursive call (flipped boolean to false)
                    max_score = max(max_score, score)

            return max_score

        # case 3: monster's turn (minimizing) (else)
            # define min score
            # evaluate position using player location
            # calc 2 different scores
                # one is minimax, the other is base score - "penalty"

            # find and return worse case