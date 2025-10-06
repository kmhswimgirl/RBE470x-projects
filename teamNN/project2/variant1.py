# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game
import random

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from qlearn_character import ApproxQLearningCharacter

for i in range(50):  # number of training games
    random.seed() # TODO Change this if you want different random choices
    print(f"\n[Training Run {i+1}/50]")
    g = Game.fromfile('map.txt')
    g.add_character(ApproxQLearningCharacter("me", "Q", 0, 0))
    g.go()

