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
        bomb = False

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
    
    def move_valid(self, wrld, new_x, new_y):
        if 0 <= new_x < wrld.width() & 0 <= new_y < wrld.height():
            if wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y):
                return True
            else:
                return False
    
    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    
    def eval_position(self,wrld, pos, target):
        # define position
        # define "score"

        # distance to exit (rescore based on dist, closer = better)

        #penalties
            # monsters
        
        # check near by the player (& keep counter?)

        # caluclate score based on "danger level" or something like that...


        # return the calculated world score

        pass
    
    def minimax_desc(self, wrld,target,depth):
        # potential moves: list[tuple(int)]

        # best move and best score

        # for new position in possible moves
            # define new_x and new_y

            # get score for this move if it is valid (use function above)

            # statement asking if caluclated move is better than the current best

         # return the best move
        pass
    
    def minimax_function(self, wrld, target, depth, is_max, player_coords):
        # case 1: player reached exit (depth == 0 || player coords == target (exit))
            # evaluate position

        # case 2: player's turn (maximizing)

        # for each possible movement
            # create new x and new y
            # confirm valid move
                # recursive call for minimax (make sure boolean is flipped!, return the max score)

        # case 3: monster's turn (minimizing) (else)
            # define min score
            # evaluate position using player location
            # calc 2 different scores
                # one is minimax, the other is base score - "penalty"

            # find and return worse case
        pass