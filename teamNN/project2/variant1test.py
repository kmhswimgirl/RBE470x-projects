# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')
sys.path.insert(1, '../teamNN')

# Import necessary stuff
from game import Game
import random

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from qlearn_character import ApproxQLearningCharacter, TRAINING
from monsters.stupid_monster import StupidMonster
from MDPwBomb import TestCharacter

random.seed() # TODO Change this if you want different random choices
g = Game.fromfile('/Users/dhruvmadan/RBE4701/RBE470x-projects/teamNN/map.txt')
agent = TestCharacter("me", "Q", 0, 0)
g.add_character(agent)
g.go(1)  # Run full game


