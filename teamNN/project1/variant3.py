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

from testcharacter2 import TestCharacter2

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter

# Create the game
# choose a random seed for consistent behavior
random.seed() # TODO Change this if you want different random choices

g = Game.fromfile('map.txt')
g.add_monster(SelfPreservingMonster("selfpreserving", # name
                                    "S",              # avatar
                                    3, 9,             # position
                                    1                 # detection range
))


g.add_character(Variant3Character("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go(1)
