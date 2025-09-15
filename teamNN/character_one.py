# This is necessary to find the main code
import sys
sys.path.insert(0, '../Bomberman')

# Import necessary stuff
from entity import CharacterEntity # type: ignore
from colorama import Fore, Back

class CharacterOne(CharacterEntity):

    def do(self, wrld):
        # Commands
        dx, dy = 0, 0
        bomb = False

        path = [
            'R','R','R','R',
            'D','D','D','D','D','D',
            'L',
            'D','D','D','D', 
            'R',
            'D','D','D','D',
            'L',
            'D','D','D','D',
            'R','R','R','R'
            ]   
        
        '''
        Hard coded path:
        ---------------
        4 R
        6 D
        1 L
        4 D
        1 R
        4 D
        1 L
        4 D
        4 R
        '''   
        if not hasattr(self, 'step'):
            self.step = 0

        if self.step < len(path): # if there is still steps left in the path
            if self.step < len(path):
                move = path[self.step]
                if move == 'U':
                    dy = -1
                elif move == 'L':
                    dx = -1
                elif move == 'D':
                    dy = 1
                elif move == 'R':
                    dx = 1
                
            new_x = self.x + dx
            new_y = self.y + dy

            if wrld.empty_at(new_x, new_y) or wrld.exit_at(new_x, new_y):
                self.step += 1
            else:
                self.step += 1
                dx, dy = 0, 0  # no moving if blocked (aka i hardcoded the path incorrectly lmao)
        else:
            dx, dy = 0, 0

        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()
        pass