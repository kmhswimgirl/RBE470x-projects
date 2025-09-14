# This is necessary to find the main code
import sys
sys.path.insert(0, '../Bomberman')
# Import necessary stuff
from entity import CharacterEntity # type: ignore
from colorama import Fore, Back

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # Your code here
        pass
