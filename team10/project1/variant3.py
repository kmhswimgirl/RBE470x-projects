# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.selfpreserving_monster import SelfPreservingMonster

# Import variant 3 character
from variant3character import Variant3Character

from testcharacterB5 import TestCharacter

# TODO This is your code!
sys.path.insert(1, '../teamNN')

# Create the game
# choose a random seed for consistent behavior
random.seed() # TODO Change this if you want different random choices

g = Game.fromfile('map.txt')
g.add_monster(SelfPreservingMonster("selfpreserving", # name
                                    "S",              # avatar
                                    3, 9,             # position
                                    1                 # detection range
))


g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go(1)
